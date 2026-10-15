"""Additional synthetic development checks, including known failure regressions."""
import json
from src.data import ROOT
from src.pipeline import Assistant

def evaluate_additional(bot=None):
    bot=bot or Assistant(enable_generation=False)
    rows=json.loads((ROOT/'data/additional_checks.json').read_text())
    for row in rows:
        out=bot(row['text'])
        cited=[s['id'] for s in out['sources'] if s['number'] in out['citations']]
        row['output']=out
        row['passed']=(row['gold'] in cited and not out['needs_human']) if row['answerable'] else out['needs_human']
    (ROOT/'evaluation/additional_checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('Additional synthetic development checks:',sum(r['passed'] for r in rows),'/',len(rows))
    return rows

if __name__=='__main__':evaluate_additional()
