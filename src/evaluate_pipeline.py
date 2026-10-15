"""Evaluate retrieval, policy drafts, review routing, and response latency."""

import json

import numpy as np
import pandas as pd

from src.data import ROOT, load
from src.pipeline import Assistant

EVAL = ROOT / "evaluation"

POLICY_QUESTIONS = [
    ("Where can I see the courier tracking link?", "en", "orders-2", True),
    (
        "My shipped order has no tracking update yet. How long can it take?",
        "en",
        "orders-2",
        True,
    ),
    ("How do I cancel an order before shipment?", "en", "cancellations-0", True),
    ("Can a shipped order still be cancelled?", "en", "cancellations-0", True),
    ("Can I change the delivery address after shipping?", "en", "shipping-3", True),
    (
        "How do I edit the address of an order that has not shipped?",
        "en",
        "shipping-3",
        True,
    ),
    (
        "My card payment failed but my balance was reduced. What happens?",
        "en",
        "payments-1",
        True,
    ),
    (
        "Both duplicate charges are completed after five business days. What should I do?",
        "en",
        "payments-2",
        True,
    ),
    ("How do I submit a new refund request?", "en", "refunds-1", True),
    (
        "Where do I get the return label after requesting a refund?",
        "en",
        "refunds-1",
        True,
    ),
    ("Where do I track the status of an existing refund?", "en", "refunds-3", True),
    ("How long does a card refund take after it is issued?", "en", "refunds-2", True),
    ("وين ألقى رابط شركة الشحن لمتابعة طلبي؟", "ar", "arabic-0", True),
    ("كم يحتاج التتبع عشان يظهر بعد ما ينشحن الطلب؟", "ar", "arabic-0", True),
    ("كيف ألغي الطلب قبل يشحن؟", "ar", "arabic-1", True),
    ("هل ينفع ألغي طلبي إذا انشحن؟", "ar", "arabic-1", True),
    ("أقدر أغير عنوان التوصيل بعد شحن الطلب؟", "ar", "arabic-2", True),
    ("وين أعدل عنوان الطلب قبل الشحن؟", "ar", "arabic-2", True),
    ("فشل الدفع لكن انخصمت الفلوس متى ينفك الحجز؟", "ar", "arabic-3", True),
    ("الخصم المكرر مكتمل من أكثر من خمسة أيام وش الحل؟", "ar", "arabic-4", True),
    ("كيف أقدم طلب استرجاع مبلغ جديد؟", "ar", "arabic-6", True),
    ("متى يجيني ملصق الإرجاع بعد تقديم الاسترداد؟", "ar", "arabic-6", True),
    ("كيف أتابع حالة استرداد قدمته قبل؟", "ar", "arabic-8", True),
    ("كم يوم يحتاج استرداد بطاقة مدى بعد إصداره؟", "ar", "arabic-7", True),
    (
        "What is the exact current GPS location of order A-12345?",
        "en",
        "",
        False,
    ),
    (
        "Please execute a refund of 500 SAR into my bank account right now.",
        "en",
        "",
        False,
    ),
    ("اعطني الموقع المباشر لمندوب طلبي الحين", "ar", "", False),
    ("نفذ استرداد 500 ريال لحسابي الآن", "ar", "", False),
]


def evaluate():
    EVAL.mkdir(exist_ok=True)
    bot = Assistant()

    questions = pd.DataFrame(
        POLICY_QUESTIONS,
        columns=["question", "lang", "gold_section", "answerable"],
    )
    questions.to_csv(ROOT / "data/policy_eval.csv", index=False)

    # Compare policy-retrieval methods on the same development questions.
    retrieval_rows = []

    for mode in ["dense", "hybrid", "expanded"]:
        for text, lang, gold, answerable in POLICY_QUESTIONS:
            if not answerable:
                continue

            found = bot.retrieve(text, mode=mode, k=5)
            ids = [section["id"] for section in found]
            rank = ids.index(gold) + 1 if gold in ids else None

            retrieval_rows.append({
                "method": mode,
                "lang": lang,
                "question": text,
                "gold_section": gold,
                "rank": rank,
                "hit_at_1": int(rank == 1),
                "hit_at_3": int(rank is not None and rank <= 3),
                "mrr_at_5": 1 / rank if rank else 0,
            })

    retrieval = pd.DataFrame(retrieval_rows)
    retrieval.to_csv(EVAL / "policy_retrieval_predictions.csv", index=False)
    retrieval.groupby(["method", "lang"])[
        ["hit_at_1", "hit_at_3", "mrr_at_5"]
    ].mean().to_csv(EVAL / "policy_retrieval_metrics.csv")

    # Matching intent is a proxy for historical-case relevance.
    test = load("test")
    case_rows = []

    for _, row in test.iterrows():
        query = bot.encode([row.text])[0]
        indices = np.argsort(-(bot.H @ query))[:5]
        relevant = bot.history.iloc[indices].intent.to_numpy() == row.intent
        positions = np.flatnonzero(relevant)

        case_rows.append({
            "ticket_id": row.ticket_id,
            "lang": row.lang,
            "hit_at_5": int(relevant.any()),
            "mrr_at_5": (
                1 / (int(positions[0]) + 1)
                if len(positions)
                else 0
            ),
        })

    cases = pd.DataFrame(case_rows)
    cases.to_csv(EVAL / "case_retrieval_predictions.csv", index=False)
    cases.groupby("lang")[["hit_at_5", "mrr_at_5"]].mean().to_csv(
        EVAL / "case_retrieval_metrics.csv"
    )

    # These inspected questions are development regression checks,
    # not an unseen final test.
    results = []

    for text, lang, gold, answerable in POLICY_QUESTIONS:
        output = bot(text)
        cited_ids = [
            section["id"]
            for section in output["sources"]
            if section["number"] in output["citations"]
        ]

        results.append({
            "question": text,
            "lang": lang,
            "answerable": answerable,
            "gold_section": gold,
            "needs_human": output["needs_human"],
            "cites_gold_section": bool(gold and gold in cited_ids),
            "latency_ms": output.get("latency_ms", 0),
            "output": output,
        })

    (EVAL / "sample_inputs_outputs.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )

    # Blank review fields are pending assessment, not passing scores.
    review = pd.DataFrame([
        {
            "question": result["question"],
            "language": result["lang"],
            "reply": result["output"]["reply"],
            "gold_section": result["gold_section"],
            "cites_gold_section": result["cites_gold_section"],
            "needs_human": result["needs_human"],
            "relevance_correct": "",
            "answer_complete": "",
            "failure_category": "",
            "reviewer_notes": "",
        }
        for result in results
    ])
    review.to_csv(EVAL / "reply_manual_review.csv", index=False)

    summaries = []

    for lang in ["en", "ar"]:
        language_results = [
            result for result in results if result["lang"] == lang
        ]
        answerable_results = [
            result for result in language_results if result["answerable"]
        ]
        unanswerable_results = [
            result for result in language_results if not result["answerable"]
        ]

        summaries.append({
            "lang": lang,
            "n": len(language_results),
            "answerable_draft_coverage": np.mean([
                not result["needs_human"]
                for result in answerable_results
            ]),
            "gold_section_citation_rate": np.mean([
                result["cites_gold_section"]
                for result in answerable_results
            ]),
            "unanswerable_human_review_rate": np.mean([
                result["needs_human"]
                for result in unanswerable_results
            ]),
            "note": (
                "Development suite inspected during debugging. "
                "Extractive policy drafts; citation matching does not "
                "prove relevance or answer completeness."
            ),
        })

    pd.DataFrame(summaries).to_csv(EVAL / "reply_metrics.csv", index=False)

    # Record request latency, excluding the first request.
    times = [result["latency_ms"] for result in results[1:]]

    (EVAL / "latency.json").write_text(
        json.dumps({
            "measurement": "warm end-to-end; first request excluded",
            "n": len(times),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "reply_mode": "extractive policy draft",
            "api_cost": 0,
            "compute_cost": "Local compute not priced.",
        }, indent=2)
    )

    tests = [
        ("cancel_en", "I want to cancel my order", "cancel_order"),
        ("cancel_ar", "أريد إلغاء طلبي", "cancel_order"),
        (
            "refund_status",
            "My refund was approved but the money has not arrived",
            "track_refund",
        ),
        ("new_refund", "I want to request a refund", "get_refund"),
        ("payment", "I was charged twice for one order", "payment_issue"),
    ]

    behavioral_rows = []

    for name, text, expected in tests:
        predicted, confidence = bot.classify(text)

        behavioral_rows.append({
            "name": name,
            "input": text,
            "expected": expected,
            "predicted": predicted,
            "passed": predicted == expected,
            "confidence": confidence,
        })

    pd.DataFrame(behavioral_rows).to_csv(
        EVAL / "behavioral_tests.csv",
        index=False,
    )

    print("Pipeline evaluation saved:", EVAL)
    return bot


if __name__ == "__main__":
    evaluate()
