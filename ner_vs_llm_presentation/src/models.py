"""Local-only model adapters. No remote LLM API or credential dependency."""
import json
import re
import time
from .evaluation import align_generated_entities, normalize_label
from .dataset import CORE_LABELS, gold_for


def build_schema(labels):
    return {
        'type':'object',
        'properties': {'entities': {'type':'array','items': {
            'type':'object',
            'properties': {'text': {'type':'string'}, 'label': {'type':'string','enum':list(labels)}},
            'required':['text','label'], 'additionalProperties':False}}},
        'required':['entities'], 'additionalProperties':False
    }


def build_prompt(text,labels,language='en',examples=()):
    categories=', '.join(labels)
    if language == 'tr':
        instructions=(f"Metinden şu kategorilere ait TÜM varlıkları çıkar: {categories}. "
                      'Yalnızca kaynak metinde birebir bulunan ifadeleri kullan. Varlık uydurma; '
                      'olumsuz cümlelerde de adı geçen varlıkları çıkar. Aynı varlık her geçtiğinde listele. '
                      'Yalnızca şu biçimde JSON döndür: {"entities":[{"text":"...","label":"..."}]}. '
                      'Ek açıklama veya Markdown kullanma. Yer adlarında/kurum adlarında kesme işaretiyle ayrılan Türkçe ekleri dahil etme.')
    else:
        instructions=(f"Extract ALL entity mentions belonging to these categories: {categories}. "
                      'Use only exact surface substrings found in the source. Do not invent or infer entities. '
                      'Extract named entities even in negated statements. Include repeated mentions separately. '
                      'Return only JSON: {"entities":[{"text":"...","label":"..."}]}. '
                      'No explanations, no Markdown. Do not include honorifics such as Dr. in person spans.')
    few=[]
    for r in examples:
        ents=[{'text':e['text'],'label':e['label']} for e in gold_for(r,labels)]
        few.append('Example input:\n'+r['text']+'\nExample output:\n'+json.dumps({'entities':ents},ensure_ascii=False))
    return instructions+'\n\n'+ '\n\n'.join(few) + ('\n\n' if few else '') + 'Source text:\n'+text


def run_spacy(text,labels=CORE_LABELS,model_name='en_core_web_sm',nlp=None):
    """NER model outputs *predefined* types; GPE/LOC normalized to LOC."""
    if nlp is None:
        import spacy
        nlp=spacy.load(model_name)
    start=time.perf_counter()
    doc=nlp(text)
    elapsed=time.perf_counter()-start
    out=[]
    for ent in doc.ents:
        label=normalize_label(ent.label_)
        if label in labels:
            out.append({'text':ent.text,'label':label,'start':ent.start_char,'end':ent.end_char})
    return {'method':'spacy','entities':out,'latency_sec':elapsed,
            'raw_json_valid':None,'clean_json_valid':None,'schema_valid':None,
            'ungrounded':[],'raw_text':None,'prompt_tokens':None,'generation_tokens':None,'error':None}


def run_ollama(text,labels=CORE_LABELS,model='qwen2.5:3b',temperature=0.,
               format_mode='schema',prompt_language='en',fewshot_examples=(),
               seed=42,host='http://localhost:11434'):
    """Calls only the local Ollama server; returns schema and anchoring diagnostics."""
    from ollama import Client
    from jsonschema import Draft202012Validator
    prompt=build_prompt(text,labels,prompt_language,fewshot_examples)
    schema=build_schema(labels)
    requested_format=schema if format_mode=='schema' else None
    if format_mode not in ('schema','prompt'):
        raise ValueError('format_mode must be schema or prompt')
    client=Client(host=host,timeout=240)
    start=time.perf_counter()
    response=client.chat(
        model=model,messages=[{'role':'user','content':prompt}],
        format=requested_format,options={'temperature':temperature,'seed':seed},stream=False,
    )
    elapsed=time.perf_counter()-start
    raw=response.message.content or ''
    raw_valid=False;clean_valid=False;schema_valid=False;data=None
    try:
        data=json.loads(raw);raw_valid=True;clean_valid=True
    except (ValueError,TypeError):
        cleaned=re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw.strip(),flags=re.IGNORECASE).strip()
        try:
            data=json.loads(cleaned);clean_valid=True
        except (ValueError,TypeError):
            pass
    if clean_valid:
        schema_valid=Draft202012Validator(schema).is_valid(data)
    items=data['entities'] if schema_valid else []
    entities,ungrounded=align_generated_entities(text,items,labels)
    return {'method':'llm','model':model,'entities':entities,'latency_sec':elapsed,
            'raw_json_valid':raw_valid,'clean_json_valid':clean_valid,'schema_valid':schema_valid,
            'ungrounded':ungrounded,'raw_text':raw,'prompt_tokens':response.prompt_eval_count,
            'generation_tokens':response.eval_count,'error':None,
            'format_mode':format_mode,'temperature':temperature,'prompt_language':prompt_language,
            'fewshot_count':len(fewshot_examples)}
