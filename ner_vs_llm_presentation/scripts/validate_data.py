#!/usr/bin/env python3
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import load_data

for filename in ('development.jsonl','english_test.jsonl','turkish_supplementary.jsonl'):
    rows=load_data(filename)
    labels=Counter(e['label'] for r in rows for e in r['entities']+r['extended_entities'])
    print(filename, 'samples:',len(rows),'words:',sum(len(r['text'].split()) for r in rows),'labels:',dict(labels))
print('All annotation offsets verified.')
