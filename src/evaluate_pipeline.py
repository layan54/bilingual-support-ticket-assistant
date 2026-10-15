"""Retrieval, answerability, sample outputs, robustness and warm end-to-end latency."""
import json,time
import numpy as np,pandas as pd
from src.data import ROOT,load
from src.pipeline import Assistant
EVAL=ROOT/'evaluation'
POLICY_QUESTIONS=[
 ('Where can I see the courier tracking link?','en','orders-2',True),
 ('My shipped order has no tracking update yet. How long can it take?','en','orders-2',True),
 ('How do I cancel an order before shipment?','en','cancellations-0',True),
 ('Can a shipped order still be cancelled?','en','cancellations-0',True),
 ('Can I change the delivery address after shipping?','en','shipping-3',True),
 ('How do I edit the address of an order that has not shipped?','en','shipping-3',True),
 ('My card payment failed but my balance was reduced. What happens?','en','payments-1',True),
 ('Both duplicate charges are completed after five business days. What should I do?','en','payments-2',True),
 ('How do I submit a new refund request?','en','refunds-1',True),
 ('Where do I get the return label after requesting a refund?','en','refunds-1',True),
 ('Where do I track the status of an existing refund?','en','refunds-3',True),
 ('How long does a card refund take after it is issued?','en','refunds-2',True),
 ('وين ألقى رابط شركة الشحن لمتابعة طلبي؟','ar','arabic-0',True),
 ('كم يحتاج التتبع عشان يظهر بعد ما ينشحن الطلب؟','ar','arabic-0',True),
 ('كيف ألغي الطلب قبل يشحن؟','ar','arabic-1',True),
 ('هل ينفع ألغي طلبي إذا انشحن؟','ar','arabic-1',True),
 ('أقدر أغير عنوان التوصيل بعد شحن الطلب؟','ar','arabic-2',True),
 ('وين أعدل عنوان الطلب قبل الشحن؟','ar','arabic-2',True),
 ('فشل الدفع لكن انخصمت الفلوس متى ينفك الحجز؟','ar','arabic-3',True),
 ('الخصم المكرر مكتمل من أكثر من خمسة أيام وش الحل؟','ar','arabic-4',True),
 ('كيف أقدم طلب استرجاع مبلغ جديد؟','ar','arabic-6',True),
 ('متى يجيني ملصق الإرجاع بعد تقديم الاسترداد؟','ar','arabic-6',True),
 ('كيف أتابع حالة استرداد قدمته قبل؟','ar','arabic-8',True),
 ('كم يوم يحتاج استرداد بطاقة مدى بعد إصداره؟','ar','arabic-7',True),
 ('What is the exact current GPS location of order A-12345?','en','',False),
 ('Please execute a refund of 500 SAR into my bank account right now.','en','',False),
 ('اعطني الموقع المباشر لمندوب طلبي الحين','ar','',False),
 ('نفذ استرداد 500 ريال لحسابي الآن','ar','',False),
]

def evaluate(generate=False):
    EVAL.mkdir(exist_ok=True)
    bot=Assistant(enable_generation=generate)
    questions=pd.DataFrame(POLICY_QUESTIONS,columns=['question','lang','gold_section','answerable'])
    questions.to_csv(ROOT/'data/policy_eval.csv',index=False)
    rows=[]
    for mode in ['dense','hybrid','expanded']:
        for q in POLICY_QUESTIONS:
            text,lang,gold,answerable=q
            if not answerable:continue
            found=bot.retrieve(text,mode=mode,k=5);ids=[x['id'] for x in found]
            rank=ids.index(gold)+1 if gold in ids else None
            rows.append({'method':mode,'lang':lang,'question':text,'gold_section':gold,'rank':rank,'hit_at_1':int(rank==1),'hit_at_3':int(rank is not None and rank<=3),'mrr_at_5':1/rank if rank else 0})
    rf=pd.DataFrame(rows);rf.to_csv(EVAL/'policy_retrieval_predictions.csv',index=False)
    rf.groupby(['method','lang'])[['hit_at_1','hit_at_3','mrr_at_5']].mean().to_csv(EVAL/'policy_retrieval_metrics.csv')
    test=load('test');case_rows=[]
    for _,r in test.iterrows():
        q=bot.encode([r.text])[0];idx=np.argsort(-(bot.H@q))[:5]
        relevant=bot.history.iloc[idx].intent.to_numpy()==r.intent
        where=np.flatnonzero(relevant)
        case_rows.append({'ticket_id':r.ticket_id,'lang':r.lang,'hit_at_5':int(relevant.any()),'mrr_at_5':1/(int(where[0])+1) if len(where) else 0})
    cases=pd.DataFrame(case_rows);cases.to_csv(EVAL/'case_retrieval_predictions.csv',index=False)
    cases.groupby('lang')[['hit_at_5','mrr_at_5']].mean().to_csv(EVAL/'case_retrieval_metrics.csv')
    # This policy suite was inspected during debugging; it is now a development regression suite, not an unseen final test.
    results=[]
    for text,lang,gold,answerable in POLICY_QUESTIONS:
        out=bot(text)
        cited=[x['id'] for x in out['sources'] if x['number'] in out['citations']]
        results.append({'question':text,'lang':lang,'answerable':answerable,'gold_section':gold,'needs_human':out['needs_human'],'cites_gold_section':bool(gold and gold in cited),'latency_ms':out.get('latency_ms',0),'output':out})
    (EVAL/'sample_inputs_outputs.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    review=pd.DataFrame([{'question':r['question'],'language':r['lang'],'reply':r['output']['reply'],'gold_section':r['gold_section'],'cites_gold_section':r['cites_gold_section'],'needs_human':r['needs_human'],'relevance_correct':'','answer_complete':'','failure_category':'','reviewer_notes':''} for r in results])
    review.to_csv(EVAL/'reply_manual_review.csv',index=False)
    summaries=[]
    for lang in ['en','ar']:
        part=[r for r in results if r['lang']==lang];ans=[r for r in part if r['answerable']];unans=[r for r in part if not r['answerable']]
        summaries.append({'lang':lang,'n':len(part),'answerable_draft_coverage':np.mean([not r['needs_human'] for r in ans]),'gold_section_citation_rate':np.mean([r['cites_gold_section'] for r in ans]),'unanswerable_human_review_rate':np.mean([r['needs_human'] for r in unans]),'note':'Development suite inspected during debugging. Extractive policy drafts; citation matching does not prove relevance or answer completeness.'})
    pd.DataFrame(summaries).to_csv(EVAL/'reply_metrics.csv',index=False)
    # Warm-up generation, then record each end-to-end request; first model-load call excluded.
    times=[r['latency_ms'] for r in results[1:]]
    (EVAL/'latency.json').write_text(json.dumps({'measurement':'warm end-to-end; first request excluded','n':len(times),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),'generation_enabled':generate,'api_cost':0,'compute_cost':'Local compute not priced.'},indent=2))
    tests=[('cancel_en','I want to cancel my order','cancel_order'),('cancel_ar','أريد إلغاء طلبي','cancel_order'),('refund_status','My refund was approved but the money has not arrived','track_refund'),('new_refund','I want to request a refund','get_refund'),('payment','I was charged twice for one order','payment_issue')]
    rows=[]
    for name,text,expected in tests:
        pred,confidence=bot.classify(text);rows.append({'name':name,'input':text,'expected':expected,'predicted':pred,'passed':pred==expected,'confidence':confidence})
    pd.DataFrame(rows).to_csv(EVAL/'behavioral_tests.csv',index=False)
    print('Pipeline evaluation saved:',EVAL)
    return bot

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--llm-selector',action='store_true',help='Optional Qwen evidence selection; default is extractive dense retrieval.')
    evaluate(generate=parser.parse_args().llm_selector)
