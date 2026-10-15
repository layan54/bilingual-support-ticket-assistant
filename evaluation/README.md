# Evaluation Results

`classification_metrics.csv` compares four classifiers on the same 264 test messages. The serving classifier was selected using validation macro-F1. The Arabic test set contains 24 messages.

The retrieval, reply, latency, and sample-output files describe the extractive pipeline evaluated in Colab. Runtime details are recorded in `basket_runtime.json`. Replies quote retrieved policy text directly.

## Retrieval and Reply Evaluation

The policy suite contains 24 answerable questions and four live-data/action requests. All 24 answerable questions received drafts citing the expected policy section, and all four live-data/action requests were routed to review.

`additional_checks.json` contains 16 synthetic development checks. All passed the policy-citation or review-routing criterion. Intent classification was assessed separately: five of the 12 answerable cases had the expected intent.

These questions were inspected during development and do not constitute an unseen final test. Historical-case relevance uses matching intent as a proxy. Policy citation checks do not establish reply correctness or completeness.

## Error Analysis

`manually_checked_failures.csv` contains ten reviewed baseline classification errors. `reviewed_pipeline_failures.csv` categorizes ten earlier reply failures. Earlier pipeline outputs are retained in `previous_run/`.

`error_analysis.csv` contains observed classification errors with provisional cause labels. Separate classifier behavioral checks passed 3/5, with failures involving refund status and duplicate payment.

Human-review fields in `reply_manual_review.csv` remain blank. Reply relevance and completeness have not yet been scored through human review.

The small Arabic test set, synthetic Arabic training data, and templated English messages limit generalization. Broader evaluation requires a separate independently authored test set.
