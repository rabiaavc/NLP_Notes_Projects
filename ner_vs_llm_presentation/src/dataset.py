"""Dataset loading and auditable span validations (stdlib only)."""
# verileri okur , etiket ve karakter konumlarını belirler ve doğrular.

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_LABELS = ('PERSON', 'ORG', 'LOC')
EXTRA_LABELS = ('DATE', 'JOB_TITLE', 'PRODUCT')
ALL_LABELS = CORE_LABELS + EXTRA_LABELS


def load_data(name='english_test.jsonl'):
    file = Path(name)
    if not file.exists():
        file = ROOT / 'data' / name
    records = [json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
    validate_records(records)
    return records


def validate_records(records):
    ids = set()
    for row in records:
        assert row['id'] not in ids, f"Duplicate ID: {row['id']}"
        ids.add(row['id'])
        assert row['language'] in ('en', 'tr')
        assert len(row['text'].split()) >= 30, f"Too short: {row['id']}"
        spans = []
        for e in row['entities'] + row['extended_entities']:
            assert e['label'] in ALL_LABELS, (row['id'], e)
            assert 0 <= e['start'] < e['end'] <= len(row['text']), (row['id'], e)
            assert row['text'][e['start']:e['end']] == e['text'], (row['id'], e)
            spans.append((e['start'], e['end']))
        spans.sort()
        assert all(a[1] <= b[0] for a,b in zip(spans,spans[1:])), f"Overlapping gold spans: {row['id']}"
    return True


def gold_for(row, labels=CORE_LABELS):
    labels=set(labels)
    return [e for e in row['entities'] + row['extended_entities'] if e['label'] in labels]
