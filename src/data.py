"""Reproducible six-intent data; no test rows enter any training/index split."""
from pathlib import Path
import hashlib, json, re, urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'config.json').read_text())
DATA = ROOT/'data'
BITEXT_URL = ('https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/'
              'resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

def canonical(text):
    text = re.sub(r'\{\{.*?\}\}', ' ID ', str(text))
    text = re.sub(r'\b[A-Za-z]*[-_]?\d[\w-]*\b', ' ID ', text)
    return re.sub(r'\W+', ' ', text.lower()).strip()

def prepare(raw_path=None):
    out = DATA/'prepared'; out.mkdir(exist_ok=True)
    if raw_path is None:
        raw_path = DATA/'bitext_raw.csv'
        if not raw_path.exists():
            print('Downloading Bitext customer-support data...')
            tmp=raw_path.with_suffix('.download')
            urllib.request.urlretrieve(BITEXT_URL,tmp);tmp.replace(raw_path)
    raw_path=Path(raw_path)
    expected=CONFIG.get('bitext_sha256')
    if expected and hashlib.sha256(raw_path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Bitext file differs from the verified snapshot; use the included prepared data or review the new version before updating the checksum.')
    raw = pd.read_csv(raw_path).rename(columns={'instruction':'text'})
    required={'text','intent','response'}
    if not required.issubset(raw): raise ValueError(f'Missing columns: {required-set(raw)}')
    raw=raw[raw.intent.isin(CONFIG['labels'])].dropna(subset=['text','intent']).copy()
    raw['text']=raw.text.astype(str).str.replace(r'\s+',' ',regex=True).str.strip()
    raw=raw[raw.text.str.len()>0]
    raw['family']=raw.text.map(canonical)
    # Different labels sharing the exact canonical message are ambiguous: drop them.
    ambiguous=raw.groupby('family').intent.nunique();bad=set(ambiguous[ambiguous>1].index)
    raw=raw[~raw.family.isin(bad)].drop_duplicates('family').reset_index(drop=True)
    raw['ticket_id']=['bitext-'+str(i) for i in range(len(raw))]
    raw['lang']='en';raw['source']='Bitext CDLA-Sharing-1.0';raw['dialect']='en'
    frames={k:[] for k in ['train','val','test','history']}
    for label in CONFIG['labels']:
        part=raw[raw.intent==label].sample(frac=1,random_state=CONFIG['seed'])
        n=CONFIG['test_per_intent'];v=CONFIG['val_per_intent'];t=CONFIG['train_per_intent']
        if len(part)<n+v+t+1: raise ValueError(f'Insufficient distinct data for {label}: {len(part)}')
        frames['test'].append(part.iloc[:n]);frames['val'].append(part.iloc[n:n+v])
        frames['train'].append(part.iloc[n+v:n+v+t]);frames['history'].append(part.iloc[n+v+t:])
    frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
    frames['history']=frames['history'].sample(n=min(CONFIG['history_size'],len(frames['history'])),random_state=CONFIG['seed']).reset_index(drop=True)
    for split in ['train','val','test']:
        ar=pd.read_csv(DATA/f'arabic_{split}.csv')
        frames[split]=pd.concat([frames[split],ar],ignore_index=True)
    # Exact canonical overlap checks across all splits, including the retrieval index.
    keys={k:set(df.text.map(canonical)) for k,df in frames.items()}
    names=list(keys)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlap=keys[a]&keys[b]
            if overlap: raise ValueError(f'Canonical overlap between {a} and {b}: {len(overlap)}')
    for split,df in frames.items(): df.to_csv(out/f'{split}.csv',index=False)
    manifest={'seed':CONFIG['seed'],'bitext_url':BITEXT_URL,'raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
              'counts':{k:df.groupby('lang').size().to_dict() for k,df in frames.items()},
              'exact_canonical_overlap':0,'near_duplicate_note':'Lexical grouping does not remove every semantic/template near-duplicate.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['counts'],indent=2));return frames

def load(split):
    path=DATA/'prepared'/f'{split}.csv'
    if not path.exists(): raise FileNotFoundError('Run the data preparation cell first.')
    return pd.read_csv(path).fillna({'response':''})

if __name__=='__main__':prepare()
