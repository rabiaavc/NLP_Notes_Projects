"""Create charts only from actual saved experiment summaries, never dummy scores."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
if not list(OUT.glob('*_summary.json')):
    raise SystemExit('No measured results yet. Run scripts/run_experiments.py first.')
import matplotlib.pyplot as plt

for file in sorted(OUT.glob('*_summary.json')):
    report=json.loads(file.read_text(encoding='utf-8'))
    names=list(report['conditions'].keys())
    vals=[report['conditions'][n]['f1'] for n in names]
    if not names: continue
    fig,ax=plt.subplots(figsize=(max(7,len(names)*1.5),4))
    bars=ax.bar(range(len(names)),vals)
    ax.set_ylim(0,1)
    ax.set_ylabel('Entity-level micro F1')
    ax.set_xticks(range(len(names)),[x.replace('__','\n') for x in names],rotation=20,ha='right')
    ax.set_title(report['experiment'].title()+' — measured results')
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.02,f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout()
    out=OUT/(report['experiment']+'_f1.png')
    fig.savefig(out,dpi=160)
    plt.close(fig)
    print(out)
