"""Strict entity-level metrics and deterministic LLM text-to-span alignment."""
import re
from collections import Counter


def normalize_label(label):
    mapping = {'PER': 'PERSON', 'PERSON': 'PERSON', 'ORG': 'ORG',
               'GPE': 'LOC', 'LOC': 'LOC', 'LOCATION': 'LOC',
               'DATE': 'DATE', 'JOB_TITLE': 'JOB_TITLE', 'PRODUCT': 'PRODUCT'}
    return mapping.get(str(label).upper())


def align_generated_entities(source_text, generated_entities, allowed_labels):
    """Map exact surface strings onto source char spans, or flag as ungrounded.

    Repeated mentions are assigned in source order. No semantic guessing or alias
    normalization is performed. 'Paris' cannot be extracted if text says 'London'.
    """
    spans=[]; ungrounded=[]; used=set(); allowed=set(allowed_labels)
    for item in generated_entities:
        if not isinstance(item,dict):
            ungrounded.append({'value':item,'reason':'not_object'}); continue
        phrase=item.get('text'); label=normalize_label(item.get('label'))
        if not isinstance(phrase,str) or not phrase.strip() or label not in allowed:
            ungrounded.append({'value':item,'reason':'invalid_type_or_label'}); continue
        if phrase != phrase.strip():
            ungrounded.append({'value':item,'reason':'leading_trailing_whitespace'}); continue
        matches=list(re.finditer(re.escape(phrase),source_text,flags=re.IGNORECASE))
        match=next((m for m in matches if (m.start(),m.end()) not in used),None)
        if match is None:
            ungrounded.append({'value':item,'reason':'not_found_or_duplicate'}); continue
        used.add((match.start(),match.end()))
        spans.append({'text':source_text[match.start():match.end()],
                      'label':label,'start':match.start(),'end':match.end()})
    return sorted(spans,key=lambda x:(x['start'],x['end'])),ungrounded


def score_entities(gold,predicted):
    """Micro exact-span typed entity score; duplicates count once per record."""
    key=lambda e:(e['start'],e['end'],e['label'])
    a=Counter(map(key,gold)); b=Counter(map(key,predicted))
    tp=sum((a&b).values()); fp=sum((b-a).values()); fn=sum((a-b).values())
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f=2*p*r/(p+r) if p+r else 0.
    return {'tp':tp,'fp':fp,'fn':fn,'precision':p,'recall':r,'f1':f}


def summarize(rows):
    """Aggregate valid per-record output; metrics are micro, not mean F1."""
    tp=sum(r['tp'] for r in rows); fp=sum(r['fp'] for r in rows); fn=sum(r['fn'] for r in rows)
    p=tp/(tp+fp) if tp+fp else 0.
    rec=tp/(tp+fn) if tp+fn else 0.
    lat=[r['latency_sec'] for r in rows if isinstance(r.get('latency_sec'),(int,float))]
    from statistics import median
    def rate(field):
        vals=[r[field] for r in rows if r.get(field) is not None]
        return sum(bool(v) for v in vals)/len(vals) if vals else None
    return {'samples':len(rows),'precision':p,'recall':rec,
            'f1':2*p*rec/(p+rec) if p+rec else 0.,
            'tp':tp,'fp':fp,'fn':fn,
            'raw_json_validity':rate('raw_json_valid'),
            'clean_json_validity':rate('clean_json_valid'),
            'schema_validity':rate('schema_valid'),
            'avg_latency_sec':sum(lat)/len(lat) if lat else None,
            'median_latency_sec':median(lat) if lat else None,
            'ungrounded_total':sum(r.get('ungrounded_count',0) for r in rows)}
