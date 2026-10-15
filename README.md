# Basket: Bilingual Support Ticket Assistant

## Problem

Classify Arabic and English e-commerce support messages, retrieve similar cases, and prepare policy-supported replies for support agents. Basket is a fictional store. The assistant cannot access customer accounts or execute transactions.

## Data

Six intents: `track_order`, `cancel_order`, `change_shipping_address`, `payment_issue`, `get_refund`, `track_refund`.

| Split | English | Arabic |
|---|---:|---:|
| Training | 1,200 | 120 |
| Validation | 240 | 36 |
| Test | 240 | 24 |
| Historical cases | 1,800 | 0 |

English messages come from Bitext. Arabic training and validation examples are synthetic and were authored separately from the Arabic test set. The bilingual policy collection supports replies. Exact canonical messages and ticket IDs are disjoint across splits; template and semantic similarity can remain. The prepared CSV files are included.

## Approach

- **Baseline:** English-only character TF-IDF + logistic regression.
- **Embedding classifiers:** Frozen multilingual e5 + logistic regression, comparing English-only and bilingual training.
- **Fine-tuning:** Multilingual DistilBERT on the bilingual split, with early stopping and serving-model selection based on validation macro-F1.
- **Retrieval:** Multilingual embeddings retrieve historical cases and policy sections. Dense search, hybrid search, and explicit query expansion are compared on the same development questions.
- **Replies:** The default draft quotes the retrieved policy and cites its source. Low-confidence, weak-evidence, and recognized unsupported requests route to review. Scope patterns do not detect all unsupported requests. All drafts require agent review.

The optional Qwen evidence selector remains available in the Python implementation but is not used by the notebook's default demo. The evaluated default uses extractive policy drafts rather than free-form generation.

## Results

Classifier comparison on the same 264 test messages:

| System | English macro-F1 | Arabic macro-F1 | Overall macro-F1 |
|---|---:|---:|---:|
| TF-IDF | 1.0000 | 0.0476 | 0.9295 |
| English-only e5 | 0.9958 | 0.6731 | 0.9697 |
| Bilingual e5 | 0.9958 | 0.9111 | 0.9886 |
| Fine-tuned DistilBERT | 0.9917 | 0.8737 | 0.9811 |

Bilingual e5 was selected on validation macro-F1 (0.9711). Fine-tuning was completed and evaluated but did not improve on that classifier. The shared Arabic test set contains only 24 messages; the bilingual e5 95% macro-F1 interval is [0.7778, 1.0000], overlapping the fine-tuned encoder interval. Results do not establish a general ranking of methods.

The extractive pipeline cited the expected policy for 24/24 answerable development questions and routed 4/4 live-data/action requests to review. The additional development checks passed 16/16, where success means citing the expected policy or routing an unsupported request to review—not predicting every intent correctly.

These sets were inspected during development and are regression checks, not an unseen final test or proof of complete answer correctness. Historical-case relevance uses matching intent as a proxy. A separate independently authored test is needed for broader claims.

Measured predictions, confidence intervals, retrieval metrics, reply metrics, latency, and sample outputs are available in `evaluation/`. Runtime details for this execution are recorded in `evaluation/basket_runtime.json`.

## Error Analysis

`evaluation/manually_checked_failures.csv` contains ten observed baseline errors with a checked vocabulary-coverage diagnosis. `evaluation/reviewed_pipeline_failures.csv` categorizes ten earlier reply failures: unnecessary escalation, policy contradiction, incomplete answer, and unsupported action claims. Suggested causes in `evaluation/error_analysis.csv` remain provisional.

The selected classifier made three errors on the main test. Separate behavioral checks passed 3/5, with failures involving refund status and duplicate payment.

An inspection of the 12 answerable additional development cases found five correct intent predictions, using the intent associated with each reference policy as the expected label. Intent accuracy was not the pass criterion for those additional checks.

Human-review fields in `evaluation/reply_manual_review.csv` remain blank until reviewed. The small Arabic test, synthetic Arabic training data, and templated English messages limit generalization.

## How to Run

1. Open [the notebook in Google Colab](https://colab.research.google.com/github/layan54/bilingual-support-ticket-assistant/blob/main/Support_Ticket_Assistant_Colab.ipynb).
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run cells in order. Training and evaluation code is shown in the notebook; running all cells retrains all classifiers. Public model downloads require internet access. No paid API key is required.
4. Launch the bilingual Gradio demo and inspect its replies and evidence. Use demo messages with the temporary public Gradio link.
5. Download the executed notebook to preserve outputs. The final cell exports project files and results; large encoder weights, caches, and certificates are excluded.

To launch the saved e5 demo locally with PyTorch installed:

```bash
python -m pip install -r requirements.txt
python app.py
```

To reproduce the experiments through the Python modules, using a suitable GPU for fine-tuning:

```bash
python -m src.train tfidf
python -m src.train embeddings
python -m src.train finetune
python -m src.evaluate
python -m src.evaluate_pipeline
python -m src.evaluate_additional
```

## Sample Inputs and Outputs

| Input | Intended behavior |
|---|---|
| `كيف ألغي الطلب قبل يشحن؟` | Classify cancellation and cite the policy explaining cancellation before shipment. |
| `Can I change the delivery address after shipping?` | Cite the policy limiting post-shipment changes to pickup-point redirection in the same city. |
| `نفذ استرداد 500 ريال لحسابي الآن` | Explain the capability limit and route the requested action to an agent. |

These describe intended behavior. Actual measured outputs, predicted intents, and cited sources are saved in `evaluation/sample_inputs_outputs.json`.

## References

- [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset): English messages and historical cases. Distributed under CDLA-Sharing-1.0; see `DATA_LICENSE.txt`.
- [Arabic ticket dataset](https://github.com/SDAIAAcademy/applied-nlp/blob/main/labs/data/arabic_tickets.csv): source of the 24-example six-intent Arabic test subset. These examples were not used for training or query expansion.
- [Policy knowledge base](https://github.com/SDAIAAcademy/applied-nlp/tree/main/labs/data/kb): source of the English policies; the project includes condensed Arabic adaptations and uses the Basket store name.
- Arabic training/validation data and additional development questions were AI-authored for this project, not collected from real customers.
- [SDAIA Academy](https://github.com/SDAIAAcademy)
