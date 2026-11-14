#!/usr/bin/env python3
"""Reproducible baseline and ablation runner. Execute from project root.

Run individual experiments, inspect result CSVs, then run the remaining ones.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.dataset import load_data,gold_for,CORE_LABELS
from src.evaluation import score_entities,summarize
from src.models import run_spacy,run_ollama

OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
EXPERIMENTS=('accuracy','flexibility','ambiguity','json','efficiency','temperature','fewshot','language','unseen','quantization')


def run_model(row, method, labels=CORE_LABELS, nlp=None, **kwargs):
    if method=='spacy':return run_spacy(row['text'],labels,nlp=nlp)
    return run_ollama(row['text'],labels,**kwargs)


def assemble(row,method,result,labels,condition):
    stats=score_entities(gold_for(row,labels),result['entities'])
    # Ungrounded/hallucinated entity strings are false positives, not free omissions.
    if result['ungrounded']:
        stats['fp'] += len(result['ungrounded'])
        tp,fp,fn=stats['tp'],stats['fp'],stats['fn']
        stats['precision']=tp/(tp+fp) if tp+fp else 0.0
        stats['recall']=tp/(tp+fn) if tp+fn else 0.0
        p,r=stats['precision'],stats['recall']
        stats['f1']=2*p*r/(p+r) if p+r else 0.0
    return {'sample_id':row['id'],'group':row['group'],'language':row['language'],
            'condition':condition,'method':method,**stats,
            'latency_sec':round(result['latency_sec'],5),
            'raw_json_valid':result['raw_json_valid'],
            'clean_json_valid':result['clean_json_valid'],
            'schema_valid':result['schema_valid'],
            'ungrounded_count':len(result['ungrounded']),
            'ungrounded_detail':json.dumps(result['ungrounded'],ensure_ascii=False),
            'prompt_tokens':result['prompt_tokens'],
            'generation_tokens':result['generation_tokens'],
            'gold':json.dumps(gold_for(row,labels),ensure_ascii=False),
            'predicted':json.dumps(result['entities'],ensure_ascii=False),
            'raw_model_response':result['raw_text'] or ''}


def main():
    ap=argparse.ArgumentParser(description='Run local Traditional NER and Ollama LLM experiments')
    ap.add_argument('--experiment',choices=EXPERIMENTS+('all',),default='accuracy')
    ap.add_argument('--limit',type=int,default=5,help='max texts per experiment; 0 means all')
    ap.add_argument('--model',default='qwen2.5:3b',help='Local Ollama model identifier')
    ap.add_argument('--q4-model',default=None)
    ap.add_argument('--q8-model',default=None)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--temperature',type=float,default=0.0)
    ap.add_argument('--repeat',type=int,default=1,help='repeats for speed/temperature')
    ap.add_argument('--no-spacy',action='store_true',help='LLM-only pilot')
    args=ap.parse_args()
    if args.limit<0 or args.repeat<1:ap.error('limit must be >= 0, repeat >= 1')
    if args.experiment=='quantization' and not (args.q4_model and args.q8_model):
        ap.error('Quantization requires --q4-model and --q8-model, from the SAME base model')

    dataset=load_data('english_test.jsonl')
    supplementary=load_data('turkish_supplementary.jsonl')
    development=load_data('development.jsonl')
    dev_en=[r for r in development if r['language']=='en']
    fewshot=dev_en[:3]
    nlp=None
    needs_spacy=args.experiment in ('accuracy','ambiguity','efficiency','unseen','flexibility','all') and not args.no_spacy
    if needs_spacy:
        import spacy
        nlp=spacy.load('en_core_web_sm')
        print('spaCy NER labels:',nlp.get_pipe('ner').labels)
    selections=EXPERIMENTS if args.experiment=='all' else (args.experiment,)
    if args.experiment=='all' and not (args.q4_model and args.q8_model):
        selections=tuple(x for x in selections if x!='quantization')
        print('Skipping optional quantization: Q4 and Q8 tags were not supplied.')
    def clipped(items): return items[:args.limit] if args.limit else items
    def llm(row,labels=CORE_LABELS,**kwargs):
        opts={'model':args.model,'seed':args.seed,'temperature':args.temperature,'format_mode':'schema'}
        opts.update(kwargs)
        return run_ollama(row['text'],labels,**opts)
    for exp in selections:
        records=[]
        def add(row,method,result,condition,labels=CORE_LABELS):
            records.append(assemble(row,method,result,labels,condition))
        if exp=='accuracy':
            for r in clipped(dataset):
                if nlp is not None:add(r,'spacy',run_spacy(r['text'],nlp=nlp),'standard')
                add(r,'llm',llm(r),'standard')
        elif exp=='flexibility':
            print('JOB_TITLE is not present in en_core_web_sm labels; report unsupported, not F1=0.')
            subset=[r for r in dataset if any(e['label']=='JOB_TITLE' for e in r['extended_entities'])]
            for r in clipped(subset):
                add(r,'llm',llm(r,labels=('JOB_TITLE',)),'new_category',('JOB_TITLE',))
        elif exp=='ambiguity':
            subset=[r for r in dataset if r['group']=='ambiguity']
            for r in clipped(subset):
                if nlp is not None:add(r,'spacy',run_spacy(r['text'],nlp=nlp),'ambiguity')
                add(r,'llm',llm(r),'ambiguity')
        elif exp=='json':
            for r in clipped(dataset):
                add(r,'llm',llm(r,format_mode='prompt'),'prompt_only')
                add(r,'llm',llm(r,format_mode='schema'),'schema_constrained')
        elif exp=='efficiency':
            # warm up both methods; do not include warm-up in timing statistics
            sample=dataset[0]
            if nlp is not None:run_spacy(sample['text'],nlp=nlp)
            llm(sample)
            for r in clipped(dataset):
                for i in range(args.repeat):
                    if nlp is not None:add(r,'spacy',run_spacy(r['text'],nlp=nlp),f'warm_repeat_{i+1}')
                    add(r,'llm',llm(r),f'warm_repeat_{i+1}')
        elif exp=='temperature':
            for r in clipped(dataset):
                for t in (0.,1.):
                    for i in range(max(args.repeat,2)):
                        add(r,'llm',llm(r,temperature=t,seed=args.seed+i),f'temp_{t}_repeat_{i+1}')
        elif exp=='fewshot':
            for r in clipped(dataset):
                add(r,'llm',llm(r,fewshot_examples=()),'zero_shot')
                add(r,'llm',llm(r,fewshot_examples=fewshot),'three_shot')
        elif exp=='language':
            # LLM-only: Turkish is SUPPLEMENTARY; use exactly EN pairs matched to TR examples.
            for tr in clipped(supplementary):
                en=next(r for r in dataset if r['pair_id']==tr['pair_id'])
                for r in (en,tr):
                    for lang in ('en','tr'):
                        add(r,'llm',llm(r,prompt_language=lang),f'prompt_{lang}_text_{r["language"]}')
        elif exp=='unseen':
            # Dev examples are not blind test samples; use separate challenge cases from a held-out test group.
            for r in clipped([r for r in dataset if r['group']=='challenging']):
                if nlp is not None:add(r,'spacy',run_spacy(r['text'],nlp=nlp),'challenge')
                add(r,'llm',llm(r),'challenge')
        elif exp=='quantization':
            for r in clipped(dataset):
                add(r,'llm',llm(r,model=args.q4_model),'q4')
                add(r,'llm',llm(r,model=args.q8_model),'q8')
        if not records:continue
        path=OUT/f'{exp}_detail.csv'
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0].keys()));w.writeheader();w.writerows(records)
        by_cond={}
        for condition in sorted({(row['method'],row['condition']) for row in records}):
            sub=[r for r in records if (r['method'],r['condition'])==condition]
            by_cond[condition[0]+'__'+condition[1]]=summarize(sub)
        report={'experiment':exp,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
                'model':args.model,'seed':args.seed,'requested_limit':args.limit,
                'notes':('Prototype results on a synthetic dataset. All scores require manual gold review. '
                         'TR is supplementary; language differences confound NER checkpoint differences. '
                         'Repeated temperature trials are not independent test examples.'),
                'conditions':by_cond}
        (OUT/f'{exp}_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'[{exp}] {len(records)} runs -> {path.relative_to(ROOT)}')
        for name,metrics in by_cond.items():
            print(f'    {name:36} n={metrics["samples"]:3d} F1={metrics["f1"]:.3f} raw_json={metrics["raw_json_validity"]} time={metrics["avg_latency_sec"]:.3f}s')
    print('Finished. Never treat unrun experiments as measured results.')

if __name__=='__main__':main()
