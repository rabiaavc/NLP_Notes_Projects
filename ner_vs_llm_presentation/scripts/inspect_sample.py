#!/usr/bin/env python3
"""Print annotated text and spans without loading any ML model."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import load_data

parser=argparse.ArgumentParser(); parser.add_argument('--id',default='EN031'); args=parser.parse_args()
for name in ('english_test.jsonl','turkish_supplementary.jsonl','development.jsonl'):
    for row in load_data(name):
        if row['id']==args.id:
            print('TEXT:',row['text'],'\nGOLD:')
            for e in sorted(row['entities']+row['extended_entities'],key=lambda e:e['start']):
                print(f"  [{e['start']:3d}:{e['end']:3d}] {e['label']:10} {e['text']}")
            sys.exit(0)
raise SystemExit('Unknown ID: '+args.id)
