"""Same-test classification comparisons, confidence intervals and categorized errors."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import CONFIG, ROOT, load

EVAL = ROOT / "evaluation"
LABELS = CONFIG["labels"]


def score(df):
    return f1_score(
        df.y_true,
        df.y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )


def ci(df, n=300):
    rng = np.random.default_rng(CONFIG["seed"])
    scores = []

    # Stratified bootstrap retains every intent in small Arabic test samples.
    groups = [group for _, group in df.groupby("y_true")]

    for _ in range(n):
        sample = pd.concat(
            [
                group.iloc[rng.integers(0, len(group), len(group))]
                for group in groups
            ]
        )
        scores.append(score(sample))

    return np.percentile(scores, [2.5, 97.5]).tolist()


def report():
    files = [
        EVAL / f"{name}_predictions.csv"
        for name in [
            "tfidf_en",
            "e5_en",
            "e5_bilingual",
            "encoder_finetuned",
        ]
        if (EVAL / f"{name}_predictions.csv").exists()
    ]

    frames = {
        path.name.removesuffix("_predictions.csv"): pd.read_csv(path)
        for path in files
    }

    if not frames:
        raise RuntimeError("No measured predictions. Run training first.")

    expected = set(load("test").ticket_id)
    rows = []
    errors = []

    for name, df in frames.items():
        if set(df.ticket_id) != expected or df.ticket_id.duplicated().any():
            raise ValueError(
                f"{name}: not the same complete held-out test set"
            )

        gold = load("test").set_index("ticket_id").intent

        if not df.y_true.equals(
            df.ticket_id.map(gold).rename("y_true")
        ):
            raise ValueError(f"{name}: gold labels mismatch")

        for lang in ["all", "en", "ar"]:
            group = df if lang == "all" else df[df.lang == lang]
            lo, hi = ci(group)

            rows.append(
                {
                    "model": name,
                    "slice": lang,
                    "n": len(group),
                    "accuracy": accuracy_score(
                        group.y_true, group.y_pred
                    ),
                    "macro_f1": score(group),
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )

        cm = confusion_matrix(
            df.y_true,
            df.y_pred,
            labels=LABELS,
        )

        pd.DataFrame(
            cm,
            index=LABELS,
            columns=LABELS,
        ).to_csv(EVAL / f"{name}_confusion.csv")

        err = df[df.y_true != df.y_pred].copy()

        # Observed failures are distinct from provisional causal diagnoses.
        err["cause"] = np.where(
            (err.lang == "ar") & name.endswith("_en"),
            "coverage: English-only training",
            "model error: incorrect intent",
        )

        err["diagnosis_status"] = np.where(
            (err.lang == "ar") & name.endswith("_en"),
            (
                "supported training-language gap; "
                "per-example review still recommended"
            ),
            "provisional; review ambiguity and gold label",
        )

        err["stage"] = "classification"
        errors.append(err)

    table = pd.DataFrame(rows)
    table.to_csv(EVAL / "classification_metrics.csv", index=False)

    failures = pd.concat(errors, ignore_index=True)
    failures.to_csv(
        EVAL / "all_categorized_failures.csv",
        index=False,
    )

    # Prioritize distinct failing messages.
    first = failures.drop_duplicates("ticket_id").head(20)
    first.to_csv(EVAL / "error_analysis.csv", index=False)

    summary = {
        "evaluation_status": (
            "partial"
            if "encoder_finetuned" not in frames
            else (
                "classifier comparison complete; "
                "inspect retrieval and reply reports too"
            )
        ),
        "models": list(frames),
        "test_size": len(expected),
        "distinct_failed_messages": len(
            failures.ticket_id.unique()
        ),
        "reviewed_failure_table_size": len(first),
        "at_least_10_observed_failures": len(first) >= 10,
        "note": (
            "Cause tags are diagnostic suggestions unless a separate "
            "manual review file confirms them. No fabricated failures."
        ),
    }

    (EVAL / "status.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(table.round(4).to_string(index=False))
    print(json.dumps(summary, indent=2))

    return table


if __name__ == "__main__":
    report()
