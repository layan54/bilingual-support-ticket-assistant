# Evaluation results

`classification_metrics.csv` contains the measured four-system classifier comparison from the user’s Colab run. Models share the same 264 test IDs. The serving model was selected on validation only; no classifier was retrained during the reply revision. The Arabic test has only 24 messages.

Current top-level retrieval, reply, latency and sample-output files describe the **revised extractive default pipeline**, executed locally on CPU using the saved e5 classifier. `colab_run/revised_runtime.json` records the uploaded Colab environment. `basket_validation.json` records the current Basket checks. `previous_run/` retains earlier generative pipeline outputs, including failures. Do not compare CPU extractive latency with GPU generative latency as if they were the same workload.

The 28 policy questions were inspected during debugging and now form a development regression suite: 24 answerable questions and 4 live-data/action requests. All 24 revised drafts cite the expected section; all 4 unsupported actions route to review. This does not prove complete answer correctness or broad out-of-scope robustness.

`additional_checks.json` contains 16 synthetic development cases, including phrasing and known unsupported topics. Once failures were inspected and fixes added, this set also became development data. Its pre-fix results are preserved under `previous_run/additional_checks_before_guards.json`.

`manually_checked_failures.csv` records ten baseline classification failures. `reviewed_pipeline_failures.csv` records ten earlier pipeline failures with categories. `error_analysis.csv` uses provisional classifier cause tags. `reply_manual_review.csv` has blank human review fields; blank means unreviewed, not correct.

The optional Qwen evidence selector has parser regression checks but was not executed in the revised local evaluation. The default evaluated mode does not generate free-form claims. All displayed policy content is copied verbatim; retrieval relevance, completeness, capability routing and policy accuracy still require review. A separate independently authored final test is needed for generalization claims.
