"""TF-IDF, English-only/bilingual e5 baselines and multilingual encoder tuning."""
import gc,json,random,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,accuracy_score
from src.data import ROOT,CONFIG,load
ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)
EVAL=ROOT/'evaluation';EVAL.mkdir(exist_ok=True)
LABELS=CONFIG['labels']

def seed_all():
    random.seed(CONFIG['seed']);np.random.seed(CONFIG['seed'])
    import torch
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(CONFIG['seed'])

def tune_threshold(y,probs,classes):
    conf=probs.max(1);correct=np.asarray(classes)[probs.argmax(1)]==np.asarray(y)
    choices=[]
    for tau in np.linspace(0,0.99,100):
        keep=conf>=tau
        if keep.sum()>=20 and correct[keep].mean()>=.90:choices.append((keep.mean(),float(tau)))
    return max(choices)[1] if choices else 1.01 # explicit abstain-all if no validation operating point

def save_predictions(name,df,probs,classes,latency_ms=None):
    result=df[['ticket_id','text','intent','lang','dialect','source']].rename(columns={'intent':'y_true'}).copy()
    result['y_pred']=np.asarray(classes)[probs.argmax(1)];result['confidence']=probs.max(1)
    result['model']=name
    if latency_ms is not None: result['latency_ms']=latency_ms
    result.to_csv(EVAL/f'{name}_predictions.csv',index=False)
    return result

def tfidf():
    train,val,test=load('train'),load('val'),load('test')
    model=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True),LogisticRegression(C=4,max_iter=2000,random_state=CONFIG['seed']))
    en=train[train.lang=='en'];model.fit(en.text,en.intent)
    probs=model.predict_proba(test.text)
    tau=tune_threshold(val.intent,model.predict_proba(val.text),model.classes_)
    joblib.dump(model,ART/'tfidf.joblib')
    save_predictions('tfidf_en',test,probs,model.classes_)
    (ART/'tfidf_meta.json').write_text(json.dumps({'tau':tau,'training':'English-only','type':'TF-IDF char ngrams + LogisticRegression'}))
    print('TF-IDF English-only baseline saved.')

def embeddings():
    seed_all()
    from sentence_transformers import SentenceTransformer
    import torch
    train,val,test=load('train'),load('val'),load('test')
    encoder=SentenceTransformer(CONFIG['embedding_model'],revision=CONFIG.get('model_revisions',{}).get('embedding_model'),device='cuda' if torch.cuda.is_available() else 'cpu')
    X={}
    for name,df in [('train',train),('val',val),('test',test)]:
        X[name]=encoder.encode(['query: '+t for t in df.text],normalize_embeddings=True,batch_size=64,show_progress_bar=True)
    metadata={}
    for name,mask in [('e5_en',train.lang.eq('en').to_numpy()),('e5_bilingual',np.ones(len(train),dtype=bool))]:
        model=LogisticRegression(C=10,max_iter=3000,random_state=CONFIG['seed']).fit(X['train'][mask],train.intent.to_numpy()[mask])
        tau=tune_threshold(val.intent,model.predict_proba(X['val']),model.classes_)
        p=model.predict_proba(X['test'])
        joblib.dump(model,ART/f'{name}.joblib');save_predictions(name,test,p,model.classes_)
        metadata[name]={'tau':tau,'n_train':int(mask.sum()),'embedding_model':CONFIG['embedding_model']}
        # Warm-up followed by wall-clock, single-request embedding + classification latency.
        for t in test.text.head(3):model.predict_proba(encoder.encode(['query: '+t],normalize_embeddings=True))
        times=[]
        for t in test.text.head(40):
            if torch.cuda.is_available():torch.cuda.synchronize()
            start=time.perf_counter();model.predict_proba(encoder.encode(['query: '+t],normalize_embeddings=True))
            if torch.cuda.is_available():torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000)
        metadata[name]['p95_ms']=float(np.percentile(times,95))
    np.savez_compressed(ART/'e5_features.npz',**X)
    # Semantic proximity is a diagnostic, not proof of duplication.
    sim=X['test']@X['train'].T;best=sim.argmax(1)
    audit=pd.DataFrame({'ticket_id':test.ticket_id,'closest_train_id':train.ticket_id.to_numpy()[best],'max_cosine':sim.max(1),'lang':test.lang})
    audit.to_csv(EVAL/'semantic_overlap_audit.csv',index=False)
    (ART/'e5_meta.json').write_text(json.dumps(metadata,indent=2))
    del encoder;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    print('English-only and bilingual e5 baselines saved.')

def finetune():
    seed_all()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments,EarlyStoppingCallback
    if not torch.cuda.is_available():raise RuntimeError('Select Runtime > Change runtime type > T4 GPU before running fine-tuning.')
    train,val,test=load('train'),load('val'),load('test')
    label2id={x:i for i,x in enumerate(LABELS)};id2label={i:x for x,i in label2id.items()}
    tok=AutoTokenizer.from_pretrained(CONFIG['encoder_model'],revision=CONFIG.get('model_revisions',{}).get('encoder_model'))
    # Length selection uses training text only; no test-dependent hyperparameters.
    lens=[len(x) for x in tok(train.text.tolist(),truncation=False)['input_ids']]
    maxlen=int(min(CONFIG['max_length_cap'],max(16,8*np.ceil(np.percentile(lens,99)/8))))
    def dataset(df):
        ds=Dataset.from_dict({'text':df.text.tolist(),'label':df.intent.map(label2id).tolist()})
        return ds.map(lambda b:tok(b['text'],truncation=True,max_length=maxlen),batched=True,remove_columns=['text'])
    tr,va,te=map(dataset,[train,val,test])
    model=AutoModelForSequenceClassification.from_pretrained(CONFIG['encoder_model'],revision=CONFIG.get('model_revisions',{}).get('encoder_model'),num_labels=len(LABELS),label2id=label2id,id2label=id2label)
    def metrics(p):
        pred=p.predictions.argmax(-1)
        return {'macro_f1':f1_score(p.label_ids,pred,labels=list(range(len(LABELS))),average='macro',zero_division=0),'accuracy':accuracy_score(p.label_ids,pred)}
    args=TrainingArguments(output_dir=str(ART/'checkpoints'),learning_rate=CONFIG['learning_rate'],num_train_epochs=CONFIG['epochs'],per_device_train_batch_size=CONFIG['train_batch_size'],per_device_eval_batch_size=32,weight_decay=.01,warmup_ratio=.1,fp16=True,eval_strategy='epoch',save_strategy='epoch',logging_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='macro_f1',save_total_limit=1,report_to='none',seed=CONFIG['seed'])
    trainer=Trainer(model=model,args=args,train_dataset=tr,eval_dataset=va,processing_class=tok,data_collator=DataCollatorWithPadding(tok),compute_metrics=metrics,callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    pv=torch.softmax(torch.tensor(trainer.predict(va).predictions),dim=-1).numpy()
    tau=tune_threshold(val.intent,pv,LABELS)
    pt=torch.softmax(torch.tensor(trainer.predict(te).predictions),dim=-1).numpy()
    save_predictions('encoder_finetuned',test,pt,LABELS)
    dest=ART/'encoder';trainer.save_model(str(dest));tok.save_pretrained(str(dest))
    pd.DataFrame(trainer.state.log_history).to_csv(EVAL/'training_log.csv',index=False)
    # Select serving model on validation only, never on final test scores.
    allscores={'encoder_finetuned':f1_score(val.intent,np.array(LABELS)[pv.argmax(1)],labels=LABELS,average='macro',zero_division=0)}
    features=np.load(ART/'e5_features.npz')
    for name in ['e5_en','e5_bilingual']:
        lr=joblib.load(ART/f'{name}.joblib');allscores[name]=f1_score(val.intent,lr.predict(features['val']),labels=LABELS,average='macro',zero_division=0)
    best=max(allscores,key=allscores.get)
    meta={'tau':tau,'max_length':maxlen,'model':CONFIG['encoder_model'],'validation_macro_f1':allscores,'serving_model':best,'n_train':len(train)}
    (ART/'encoder_meta.json').write_text(json.dumps(meta,indent=2))
    (EVAL/'validation_selection.json').write_text(json.dumps(meta,indent=2))
    print('Best serving classifier on validation:',best,allscores)
    del trainer,model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tfidf','embeddings','finetune']);args=p.parse_args()
    globals()[args.stage]()
