"""Classify messages, retrieve cases and policies, and prepare cited policy drafts."""

import json
import re
import time

import joblib
import numpy as np

from src.data import ROOT, CONFIG, load

ART = ROOT / "artifacts"

AR_LABELS = {
    "track_order": "تتبّع الطلب",
    "cancel_order": "إلغاء الطلب",
    "change_shipping_address": "تعديل عنوان الشحن",
    "payment_issue": "مشكلة دفع",
    "get_refund": "طلب استرداد",
    "track_refund": "تتبّع الاسترداد",
}


def normalize_search(text):
    text = re.sub(r"[\u064b-\u0652\u0670]", "", text).replace("ـ", "")
    return re.sub("[أإآ]", "ا", text).replace("ى", "ي").lower()


def words(text):
    return re.findall(r"\w+", normalize_search(text))


def sections():
    output = []

    for path in sorted((ROOT / "data/kb").glob("*.md")):
        content = path.read_text()
        title = content.splitlines()[0].lstrip("# ")

        for index, part in enumerate(
            re.split(r"^## ", content, flags=re.M)[1:]
        ):
            heading, _, body = part.partition("\n")

            if body.strip():
                output.append({
                    "id": f"{path.stem}-{index}",
                    "source": f"data/kb/{path.name}",
                    "heading": heading,
                    "text": body.strip(),
                    "lang": "ar" if path.stem == "arabic" else "en",
                    "index_text": title + " > " + heading + "\n" + body.strip(),
                })

    return output


def reciprocal_rank_fusion(*rankings):
    scores = {}

    for ranking in rankings:
        for rank, index in enumerate(ranking, 1):
            scores[index] = scores.get(index, 0) + 1 / (60 + rank)

    return sorted(scores, key=scores.get, reverse=True)


def policy_excerpt(context, lang, number=1):
    """Prepare a draft using the policy text verbatim."""
    section = context[number - 1]
    intro = (
        "بحسب سياسة Basket:"
        if lang == "ar"
        else "According to Basket policy:"
    )

    return {
        "reply": intro + "\n\n" + section["text"],
        "citations": [number],
        "evidence_quote": section["text"],
        "needs_human": False,
        "draft_method": "extractive policy draft",
        "requires_agent_review": True,
    }


def capability_limit(text, lang):
    """Identify known requests requiring live data or account actions."""
    text = normalize_search(text)

    live_patterns = [
        r"\b(gps|live location|exact current.*location)\b",
        r"الموقع (المباشر|الحالي|الفعلي)",
        r"وين.*(المندوب|السائق).*الحين",
    ]

    action_patterns = [
        r"\b(execute|process|issue)\b.*\b(refund|cancellation)\b.*\b(now|right now)\b",
        r"^(please )?cancel my order (now|right now)",
        r"^(نفذ|نفذي|الغ|الغي|حول|حولي|ارجع|رجع).*?(الان|الحين|لحسابي)",
    ]

    if any(re.search(pattern, text) for pattern in live_patterns):
        return "live_data_unavailable"

    if any(re.search(pattern, text) for pattern in action_patterns):
        return "account_action_unavailable"

    return None


def expand_policy_query(text):
    """Expand explicit policy wording without changing classifier predictions."""
    normalized = normalize_search(text)

    if (
        re.search(r"الغ|كنسل", normalized)
        and not re.search(r"رسوم|تكلف", normalized)
    ):
        return "كيف ألغي الطلب قبل الشحن وبعد الشحن؟"

    if re.search(r"\bcancel", normalized):
        return text + " cancellation before after shipment"

    if (
        re.search(r"استرداد|استرجاع|refund", normalized)
        and re.search(
            r"ما وصلت|ما وصل|ما رجع|تعدت|معتمد|تم الاسترداد|approved|issued|credited",
            normalized,
        )
    ):
        return (
            text
            + " متابعة استرداد سابق لم يصل البنك"
            + " tracking existing refund status bank"
        )

    return text


def scope_limit(text):
    """Detect known unsupported topics; these patterns are not exhaustive."""
    text = normalize_search(text)

    patterns = [
        r"\b(weather|temperature|forecast|job|jobs|career|vacanc\w*|password|iphone price)\b",
        r"طقس|حراره الجو|وظيف|وظايف|وظائف|توظيف|كلمه (المرور|السر)|رمز الدخول",
        r"(كم سعر|بكم|سعر ايفون|اسعار المنتجات)",
        r"\b(price|cost)\b.*\b(iphone|laptop|product)\b",
    ]

    if any(re.search(pattern, text) for pattern in patterns):
        return "outside_supported_intents"

    return None


class Assistant:

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi

        self.embedder = SentenceTransformer(
            CONFIG["embedding_model"],
            revision=CONFIG.get("model_revisions", {}).get("embedding_model"),
            device="cpu",
        )

        meta_path = ART / "encoder_meta.json"
        meta = (
            json.loads(meta_path.read_text())
            if meta_path.exists()
            else {"serving_model": "e5_bilingual"}
        )

        self.name = meta["serving_model"]

        if self.name == "encoder_finetuned":
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )

            self.clf = AutoModelForSequenceClassification.from_pretrained(
                ART / "encoder"
            ).eval()
            self.clf_tok = AutoTokenizer.from_pretrained(ART / "encoder")
            self.maxlen = meta["max_length"]
            self.tau = meta["tau"]
        else:
            self.clf = joblib.load(ART / f"{self.name}.joblib")
            self.tau = json.loads(
                (ART / "e5_meta.json").read_text()
            )[self.name]["tau"]

        self.history = load("history").reset_index(drop=True)
        self.H = self.encode(self.history.text.tolist(), "passage: ")

        self.kb = sections()
        self.K = self.encode(
            [section["index_text"] for section in self.kb],
            "passage: ",
        )

        self.bm25 = BM25Okapi(
            [words(section["index_text"]) for section in self.kb]
        )

    def encode(self, texts, prefix="query: "):
        return self.embedder.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )

    def classify(self, text, q=None):
        if self.name == "encoder_finetuned":
            import torch

            enc = self.clf_tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.maxlen,
            )

            with torch.inference_mode():
                probabilities = torch.softmax(
                    self.clf(**enc).logits,
                    dim=-1,
                )[0].numpy()

            index = int(probabilities.argmax())
            return (
                self.clf.config.id2label[index],
                float(probabilities[index]),
            )

        features = (
            q.reshape(1, -1)
            if q is not None
            else self.encode([text])
        )
        probabilities = self.clf.predict_proba(features)[0]

        return (
            self.clf.classes_[probabilities.argmax()],
            float(probabilities.max()),
        )

    def retrieve(self, text, q=None, mode="expanded", k=3):
        if mode == "expanded":
            expanded = expand_policy_query(text)
            q = (
                self.encode([expanded])[0]
                if expanded != text or q is None
                else q
            )
        else:
            q = self.encode([text])[0] if q is None else q

        dense = np.argsort(-(self.K @ q)).tolist()
        scores = self.bm25.get_scores(words(text))

        # Exclude BM25 results with no positive keyword score.
        sparse = [
            int(index)
            for index in np.argsort(-scores)
            if scores[index] > 0
        ]

        order = (
            dense
            if mode in {"dense", "expanded"}
            else reciprocal_rank_fusion(dense[:12], sparse[:12])
        )

        lang = "ar" if re.search(r"[\u0600-\u06ff]", text) else "en"
        same_language = [
            index
            for index in order
            if self.kb[index]["lang"] == lang
        ]
        selected = (
            same_language if len(same_language) >= k else order
        )[:k]

        return [
            {
                **self.kb[index],
                "cosine": float(self.K[index] @ q),
            }
            for index in selected
        ]

    @staticmethod
    def fallback(lang, reason):
        reply = (
            "أحتاج مراجعة موظف الدعم لتأكيد الإجابة. "
            "لا أستطيع التحقق من حالة طلبك أو تنفيذ إجراءات على حسابك."
            if lang == "ar"
            else (
                "A support agent needs to review this request. "
                "I cannot check live order status or make changes "
                "to your account."
            )
        )

        return {
            "reply": reply,
            "citations": [],
            "evidence_quote": "",
            "needs_human": True,
            "reason": reason,
            "draft_method": "human-review fallback",
        }

    def __call__(self, text):
        start = time.perf_counter()
        text = str(text).strip()
        lang = "ar" if re.search(r"[\u0600-\u06ff]", text) else "en"

        if not text or len(text) > 4000:
            return {
                **self.fallback(lang, "empty_or_long_input"),
                "intent": None,
                "similar_cases": [],
                "sources": [],
            }

        q = self.encode([text])[0]
        intent, confidence = self.classify(text, q)

        indices = np.argsort(-(self.H @ q))[:3]
        cases = [
            {
                "ticket_id": self.history.iloc[index].ticket_id,
                "text": self.history.iloc[index].text,
                "intent": self.history.iloc[index].intent,
                "response": str(self.history.iloc[index].response),
                "score": round(float(self.H[index] @ q), 3),
            }
            for index in indices
        ]

        context = self.retrieve(text, q)
        capability = capability_limit(text, lang) or scope_limit(text)

        if capability:
            reply = self.fallback(lang, capability)
        elif confidence < self.tau:
            reply = self.fallback(lang, "low_classifier_confidence")
        elif not context or context[0]["cosine"] < 0.65:
            reply = self.fallback(lang, "weak_policy_match")
        else:
            reply = policy_excerpt(context, lang)

        return {
            "intent": intent,
            "intent_ar": AR_LABELS[intent],
            "confidence": round(confidence, 4),
            "threshold": self.tau,
            "classifier": self.name,
            "language": lang,
            **reply,
            "route": (
                "human_review"
                if reply["needs_human"]
                else "draft_for_agent"
            ),
            "similar_cases": cases,
            "sources": [
                {
                    "number": number,
                    "id": section["id"],
                    "source": section["source"],
                    "heading": section["heading"],
                    "text": section["text"],
                }
                for number, section in enumerate(context, 1)
            ],
            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                1,
            ),
        }
