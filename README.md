# Comment Moderation — 4-Class NLP Pipeline

Classifies user comments into four moderation categories, including a rare **threat** class (2.8% of the data). Built during a research internship at BIT Mesra (May–Jul 2026).

**Result:** out-of-fold macro-F1 **0.836** on 198k labelled comments (5-fold stratified CV), threat-class F1 **0.65**.

All code and outputs are in [`comment_moderation_pipeline.ipynb`](comment_moderation_pipeline.ipynb).

## Data

198,000 training comments and 102,000 unlabelled test comments. Each row has the comment text, posting date, thread id, three emoticon flags, up/down votes, two platform-internal numeric features (`if_1`, `if_2`), identity flags (race, religion, gender, disability) and the target `label` (0–3).

| Class | Share |
|---|---|
| 0 | 57.7% |
| 1 | 8.0% |
| 2 | 31.5% |
| 3 (threat) | 2.8% |

The dataset is not redistributed here. Put `train.csv` and `test.csv` next to the notebook to run it.

## Pipeline

1. **Text normalisation** — lower-casing, contraction expansion, URL and symbol stripping.
2. **Sparse text features** — word 1–2-gram TF-IDF (50,000 features) + character n-gram TF-IDF (30,000).
3. **36 engineered signals** — threat / identity / toxicity / neutral lexicon counts and ratios, intent regexes (`will kill`, `should die`, `all X are`, `go back`), vote-based controversy ratio and net score, caps ratio, length, and bucketed `if_1` / `if_2` values found during EDA — plus one-hot identity columns.
4. **Feature matrix** — everything stacked into one 80,058-column sparse matrix.
5. **Models (5-fold stratified CV, out-of-fold predictions)**
   - Multinomial Naive Bayes (baseline)
   - L1 Logistic Regression, `C` tuned by grid search, class weights for the rare classes
   - LightGBM with class weights
6. **Blend** — LightGBM and LR probabilities mixed with a weight found by simulated annealing on OOF F1 (LGB 0.52 / LR 0.48).
7. **Per-class thresholds** — multipliers on each class probability tuned with Nelder-Mead to maximise macro-F1.

## Results (out-of-fold macro-F1)

| Model | Class 0 | Class 1 | Class 2 | Class 3 (threat) | Macro |
|---|---:|---:|---:|---:|---:|
| Naive Bayes | 0.800 | 0.532 | 0.615 | 0.228 | 0.544 |
| Logistic Regression | 0.959 | 0.784 | 0.886 | 0.657 | 0.822 |
| LightGBM | 0.962 | 0.795 | 0.885 | 0.651 | 0.823 |
| **LGB + LR blend + thresholds** | | | | | **0.836** |

What moved the score: the TF-IDF + engineered-feature matrix took a linear model from 0.54 to 0.82; blending two different model families added another +0.013; per-class thresholds added +0.001.

## Run

```bash
pip install -r requirements.txt
jupyter notebook comment_moderation_pipeline.ipynb
```
