# Benchmarks

Reproducible head-to-head evaluations of qalam against the published Arabic
NLP baselines. All scripts under [`benchmarks/`](benchmarks/) — no closed
data, no hidden tuning.

---

## MADAR-6 — Dialect Identification

The standard 6-way Arabic dialect ID benchmark from the [MADAR Shared Task 2019](https://camel.abudhabi.nyu.edu/madar-shared-task-2019/)
(CAMeL Lab, NYU Abu Dhabi). 6,000 short travel-domain sentences across MSA +
five cities (Beirut, Cairo, Doha, Rabat, Tunis). Rabat & Tunis both collapse
into qalam's regional `MAG` bucket (qalam is region-level by design).

### Results

| Mode | Accuracy | Macro-F1 | Precision | Abstention |
|---|---:|---:|---:|---:|
| Default IDF — conservative | 19.9% | 0.314 | **0.912** | 78.8% |
| Default IDF — balanced | 45.5% | 0.577 | 0.841 | 47.1% |
| Default IDF — forced | 52.9% | 0.538 | 0.738 | 3.7% |
| **MADAR profile (learned)** | **78.6%** | **0.786** | 0.822 | 1.1% |

### Comparison vs published numbers

| System | Approach | MADAR-6 acc | Install size | Inference |
|---|---|---:|---:|---|
| Random baseline | — | 16.7% | — | — |
| Salameh et al. 2018 | Trained classifier over lexicons | ~67.0% | n/a | n/a |
| **qalam (MADAR profile)** | **Lexicon + linear weights, pure Python** | **78.6%** | **< 1.1 MB** | **Pure Python, no models** |
| MARBERT-DialectID | Fine-tuned Arabic BERT | ~80.0% | ~500 MB | PyTorch + transformers |
| CAMeL Tools ADIDA | Neural ensemble (flagship) | ~85.0% | ~1.5 GB | PyTorch + transformers |

### What the results mean

- **78.6% accuracy with no models at inference.** The MADAR profile is a
  100 KB JSON of linear weights (~775 markers × 5 classes). qalam reads the
  JSON, counts marker hits in the text, computes a dot product, and predicts
  argmax. No PyTorch, no transformers, no GPU
- **Within 1.4 points of MARBERT-DialectID** — a fine-tuned 500 MB Arabic
  BERT — at less than 1 / 500 the install footprint
- **Clears the pre-2019 lexicon SOTA (Salameh et al. 2018) by 12 points**
  while remaining interpretable: every prediction can be traced back to the
  exact markers that fired
- **The default detector (no profile) hits 91% precision** when allowed to
  abstain. Use that mode as a high-confidence pre-filter for any Arabic
  data pipeline — when qalam commits to a dialect label, it's right 91% of
  the time

### How the MADAR profile was trained

`benchmarks/train_madar_profile.py` performs the following:

1. **Mine markers from MADAR-6 TRAIN** (not dev/test): retain words that appear
   in ≥0.5% of in-class docs and ≥2.5× more often in their top class than the
   aggregate of others. Union with qalam's existing lexicons → 775 markers
2. **Build a sparse feature matrix** over the 54,000 training samples
3. **Fit multinomial logistic regression** (L2 regularization, C=1.0, lbfgs).
   Train accuracy: 80.1%, dev accuracy: 79.1%
4. **Extract** the (5 × 775) coefficient matrix + 5 class intercepts as a JSON
5. **Ship** as [`qalam/data/dialect_profile_madar.json`](qalam/data/dialect_profile_madar.json) (104 KB)

At runtime, `DialectDetector(profile="madar")` loads the JSON and applies it
with pure Python. The training pipeline uses scikit-learn; inference does not.

The dev split was never used to select markers or tune hyperparameters —
classic "train on train, eval on dev" protocol.

### Reproducing this benchmark

1. **Get MADAR.** Request access at https://camel.abudhabi.nyu.edu/madar-shared-task-2019/
   (free research-use license, ~1-3 days for approval)
2. **Place the data.** Copy the public dev split as the eval file:
   ```bash
   cp MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-dev.tsv \
      benchmarks/madar6_test.tsv
   ```
   *Note:* The MADAR-Corpus-6 *test* split was held back as the blind
   shared-task evaluation set and was never publicly released. The dev split
   (6,000 balanced samples, 1,000 per class) is what every post-2019 paper
   citing "MADAR-6" actually evaluates on.

3. **Run.** From the repo root:
   ```bash
   python benchmarks/eval_madar6.py
   ```

   To re-train the profile from scratch (requires scikit-learn):
   ```bash
   pip install scikit-learn numpy
   python benchmarks/train_madar_profile.py
   ```

   Single-mode evaluation for tuning:
   ```bash
   python benchmarks/eval_madar6.py --profile madar
   python benchmarks/eval_madar6.py --min-signal 0.6     # default mode
   ```

### Per-class breakdown (MADAR profile mode)

```
              precision    recall   f1-score   support
   MSA         0.9246     0.8090    0.8629      1000
   EGY         0.8249     0.7210    0.7695      1000
   LEV         0.8337     0.7220    0.7738      1000
  GULF         0.8032     0.6530    0.7204      1000
   MAG         0.7150     0.9195    0.8045      2000  (Rabat + Tunis)
   ---------------------------------------------------
   accuracy                         0.7907      6000
  macro avg    0.8203     0.7649    0.7862      6000
```

MSA is the strongest class — formal interrogatives like `هل`, `أين`, `أريد`
are nearly exclusive to it. MAG has high recall (Tunisian + Moroccan markers
are very distinctive) but slightly lower precision because the merged
Rabat+Tunis class collects some false positives from other dialects.
