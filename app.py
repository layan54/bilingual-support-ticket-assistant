"""Bilingual agent workspace. Run after training: python app.py."""
from html import escape
import gradio as gr
from src.pipeline import Assistant

CSS = """
.gradio-container {max-width:1180px !important; margin:auto; font-family:Arial,sans-serif;}
#hero {padding:30px 32px; border-radius:20px; background:linear-gradient(120deg,#102c46,#14575a); color:white; margin-bottom:20px;}
#hero h1 {color:white; font-size:30px; margin:8px 0;}
#hero p {color:#d7e9ec; max-width:780px; line-height:1.7;}
.eyebrow {font-size:12px; letter-spacing:2px; color:#9fe0cf; font-weight:700;}
.panel {border:1px solid #dde5ec; border-radius:16px; padding:20px; background:var(--block-background-fill);}
#reply textarea, #ticket textarea {unicode-bidi:plaintext; line-height:1.9; font-size:16px;}
#analyze {min-height:48px; border-radius:10px;}
.summary {display:flex; flex-wrap:wrap; gap:12px; margin:0 0 15px;}
.metric {flex:1; min-width:140px; border:1px solid #dae5ea; border-radius:12px; padding:14px;}
.metric small {display:block; color:#607482; margin-bottom:8px;}
.metric strong {font-size:17px; color:#155e63;}
.case {border:1px solid #dde5ec; border-radius:12px; padding:15px; margin:10px 0;}
.case p {unicode-bidi:plaintext; line-height:1.8; margin:8px 0;}
.source {border-left:3px solid #259b92; padding:12px 18px; margin:12px 0; background:var(--block-background-fill);}
.source p {unicode-bidi:plaintext; line-height:1.9;}
.footnote {font-size:12px; color:#657884; line-height:1.7;}
"""


def present(out):
    """Escape all input/model text before placing it into HTML."""
    intent=escape(out.get('intent_ar','')+' / '+str(out.get('intent') or '—'))
    route='مراجعة موظف / Human review' if out['needs_human'] else 'مسودة سياسة / Policy draft'
    summary='<div class="summary">'+''.join(
        f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>'
        for label,value in [('التصنيف · Intent',intent),('ثقة التصنيف · Confidence',f"{out.get('confidence',0):.1%}"),('المسار · Route',route)]
    )+'</div>'
    case_html=''.join(f'<div class="case"><small>{escape(c["intent"])} · Similarity {c["score"]:.3f}</small><p dir="auto">{escape(c["text"])}</p></div>' for c in out['similar_cases']) or '<p>No similar cases.</p>'
    source_html=''.join(f'<div class="source"><strong>[{c["number"]}] {escape(c["heading"])}</strong><p dir="auto">{escape(c["text"])}</p><small>{escape(c["source"])}</small></div>' for c in out['sources']) or '<p>No policy evidence.</p>'
    reply=out['reply']
    if out['citations']:reply+='\n\n'+ ' '.join(f'[{c}]' for c in out['citations'])
    reasons={'live_data_unavailable':'يتطلب بيانات مباشرة غير متاحة · Live account data is unavailable', 'account_action_unavailable':'يتطلب تنفيذ إجراء بواسطة موظف · An agent must perform this action', 'outside_supported_intents':'خارج مواضيع الدعم الستة · Outside the six supported topics', 'low_classifier_confidence':'تصنيف الرسالة غير مؤكد · Intent needs confirmation', 'weak_policy_match':'لم نجد سياسة مطابقة بما يكفي · No sufficiently relevant policy', 'empty_or_long_input':'أدخل رسالة من 1 إلى 4000 حرف · Enter 1–4000 characters', 'selector_no_evidence':'لم يتم اختيار دليل كافٍ · No sufficient evidence selected'}
    status=(reasons.get(out.get('reason'), 'تحتاج الرسالة مراجعة موظف · Agent review needed') if out['needs_human'] else 'مقتطف من السياسة للمراجعة قبل الإرسال · Policy excerpt: review before sending')
    return summary,reply,status,case_html,source_html,out


def build_app(assistant=None):
    assistant=assistant or Assistant(enable_generation=False)
    def run(text):return present(assistant(text))
    with gr.Blocks(title='Basket | Support Workspace',theme=gr.themes.Soft(primary_hue='teal',secondary_hue='slate'),css=CSS) as demo:
        gr.HTML('<div id="hero"><span class="eyebrow">BASKET · SUPPORT WORKSPACE</span><h1>مساعد خدمة العملاء</h1><p>Arabic & English ticket understanding, similar cases, and policy evidence — in one workspace.<br>حلّل رسالة العميل وراجع مسودة الرد ومصادرها.</p></div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=4,min_width=300,elem_classes='panel'):
                gr.Markdown('### رسالة العميل · Customer message')
                ticket=gr.Textbox(label='اكتب الرسالة بالعربي أو الإنجليزي',placeholder='مثال: أقدر ألغي طلبي بعد ما انشحن؟',lines=5,max_lines=10,elem_id='ticket')
                button=gr.Button('تحليل الرسالة · Analyze ticket',variant='primary',elem_id='analyze')
                gr.Examples(examples=[['كيف ألغي الطلب قبل يشحن؟'],['انخصم المبلغ مرتين لنفس الطلب'],['كيف أتابع حالة استرداد قدمته؟'],['Can I change my delivery address after shipping?']],inputs=ticket,label='جرّب مثالًا · Try an example')
                gr.HTML('<p class="footnote">Basket متجر افتراضي. المساعد يعرض إرشادات عامة ولا يدخل حساب العميل أو ينفذ معاملات.<br>Classification confidence measures the intent prediction, not reply correctness.</p>')
            with gr.Column(scale=7,min_width=360,elem_classes='panel'):
                summary=gr.HTML('<p>النتائج تظهر هنا بعد التحليل · Analyze a message to see results.</p>')
                reply=gr.Textbox(label='مسودة الرد · Evidence-based draft',lines=7,interactive=False,elem_id='reply')
                status=gr.Textbox(label='ملاحظة المراجعة · Review note',interactive=False)
                with gr.Tabs():
                    with gr.Tab('السياسات · Policy evidence'):sources=gr.HTML()
                    with gr.Tab('حالات مشابهة · Similar cases'):
                        gr.Markdown('Retrieved historical messages; their original responses are not policy authority.')
                        cases=gr.HTML()
                with gr.Accordion('تفاصيل تقنية · Technical details',open=False):raw=gr.JSON()
        outputs=[summary,reply,status,cases,sources,raw]
        button.click(run,inputs=ticket,outputs=outputs,concurrency_limit=1)
        ticket.submit(run,inputs=ticket,outputs=outputs,concurrency_limit=1)
    return demo

if __name__=='__main__':build_app().queue().launch()
