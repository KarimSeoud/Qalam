"""
Train a learned-weight profile for qalam's dialect detector using MADAR-6 train.

This script is used ONCE to produce `qalam/data/dialect_profile_madar.json`.
At inference time, qalam reads that JSON and uses the per-(marker, dialect)
weights instead of the default IDF heuristic. No sklearn dependency at runtime.

Method:
  1. Mine high-discrimination markers from MADAR-6 train (≥0.5% in-class rate,
     ≥2.5× more frequent in target class than aggregate of others)
  2. Build a binary feature matrix: rows = training docs, cols = mined markers
     + the existing qalam lexicon markers, value = 1 if marker fired, else 0
  3. Train multinomial logistic regression with L2 regularization
  4. Extract (n_classes × n_features) coefficient matrix + intercepts
  5. Save as JSON for runtime use

The JSON is read by qalam.dialect when `DialectDetector(profile="madar")`.
The default profile (`profile=None`) keeps the original IDF behavior, so this
file does not affect non-MADAR users at all.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Path setup so this script runs without installing qalam
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qalam.dialect import _DIALECT_ENTRIES, _LEXICONS, _compile_entry, _pre_normalize

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_PATH = Path(__file__).parent.parent / "MADAR-SHARED-TASK-final-release-25Jul2019" / "MADAR-Shared-Task-Subtask-1" / "MADAR-Corpus-6-train.tsv"
DEV_PATH = Path(__file__).parent.parent / "MADAR-SHARED-TASK-final-release-25Jul2019" / "MADAR-Shared-Task-Subtask-1" / "MADAR-Corpus-6-dev.tsv"
OUTPUT_PATH = Path(__file__).parent.parent / "qalam" / "data" / "dialect_profile_madar.json"

LABEL_MAP = {"MSA": "MSA", "BEI": "LEV", "CAI": "EGY", "DOH": "GULF", "RAB": "MAG", "TUN": "MAG"}
CLASSES = ["MSA", "EGY", "LEV", "GULF", "MAG"]

# Minimum statistics for a marker to be included
MIN_IN_CLASS_COUNT = 30      # marker must appear at least this many times in its top class
MIN_IN_CLASS_RATE = 0.005    # ≥0.5% of in-class documents
MIN_DISCRIMINATION = 2.5     # ≥2.5× more frequent in top class vs aggregate of others
MAX_FEATURES = 800           # cap features to avoid overfitting


# ---------------------------------------------------------------------------
# Step 1: Mine markers from TRAIN
# ---------------------------------------------------------------------------

def load_tsv(path: Path) -> list[tuple[str, str]]:
    """Returns [(text, qalam_label), ...]."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 2:
                continue
            text, madar_label = cols
            if madar_label not in LABEL_MAP:
                continue
            out.append((_pre_normalize(text), LABEL_MAP[madar_label]))
    return out


def mine_markers(train: list[tuple[str, str]]) -> list[str]:
    """Surface words with strong per-class discrimination."""
    docfreq: dict[str, Counter] = defaultdict(Counter)
    total = Counter()
    for text, cls in train:
        total[cls] += 1
        for w in set(text.split()):
            docfreq[cls][w] += 1

    candidates: set[str] = set()
    for cls in CLASSES:
        n = total[cls]
        for word, c in docfreq[cls].items():
            # Skip if not Arabic
            if not any("؀" <= ch <= "ۿ" for ch in word):
                continue
            # Skip very short words (1-char particles dominate noise)
            if len(word) < 2:
                continue
            others = sum(docfreq[other][word] for other in CLASSES if other != cls)
            others_n = sum(total[other] for other in CLASSES if other != cls)
            in_class_rate = c / n
            other_rate = others / others_n if others_n else 0
            if c < MIN_IN_CLASS_COUNT or in_class_rate < MIN_IN_CLASS_RATE:
                continue
            if other_rate > 0 and (in_class_rate / other_rate) < MIN_DISCRIMINATION:
                continue
            candidates.add(word)

    # Also include all currently-known qalam markers
    for words in _LEXICONS.values():
        for w in words:
            candidates.add(w)

    # Sort for determinism
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Step 2: Feature extraction
# ---------------------------------------------------------------------------

def build_patterns(markers: list[str]) -> list[re.Pattern]:
    return [_compile_entry(m) for m in markers]


def featurize(
    texts: list[str],
    markers: list[str],
    patterns: list[re.Pattern],
) -> np.ndarray:
    """Returns float32 matrix [n_texts, n_markers] with counts (clipped at 3)."""
    n = len(texts)
    k = len(markers)
    X = np.zeros((n, k), dtype=np.float32)
    for i, text in enumerate(texts):
        for j, pat in enumerate(patterns):
            hits = len(pat.findall(text))
            if hits:
                # Clip — multiple hits of the same marker shouldn't dominate
                X[i, j] = min(hits, 3)
    return X


# ---------------------------------------------------------------------------
# Step 3: Train + extract weights
# ---------------------------------------------------------------------------

def main() -> None:
    if not TRAIN_PATH.exists():
        print(f"✗ TRAIN file not found at {TRAIN_PATH}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading MADAR-6 train from {TRAIN_PATH.name}...")
    train = load_tsv(TRAIN_PATH)
    print(f"  {len(train):,} training samples across {len(set(c for _, c in train))} classes")

    print("\nMining markers...")
    markers = mine_markers(train)
    print(f"  {len(markers)} markers (qalam lexicon + mined)")
    if len(markers) > MAX_FEATURES:
        print(f"  capping to top {MAX_FEATURES} by training frequency")
        freq = Counter()
        for text, _ in train:
            for w in set(text.split()):
                if w in markers:
                    freq[w] += 1
        markers = sorted([m for m, _ in freq.most_common(MAX_FEATURES)])

    print("\nBuilding feature matrix...")
    patterns = build_patterns(markers)
    X_train = featurize([t for t, _ in train], markers, patterns)
    y_train = np.array([CLASSES.index(c) for _, c in train])
    print(f"  X_train: {X_train.shape}, density: {(X_train > 0).mean():.3f}")

    print("\nTraining logistic regression (L2, multinomial)...")
    # L2 with moderate C to prevent overfitting
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    print(f"  train accuracy: {train_acc:.4f}")

    # Eval on dev for sanity check
    if DEV_PATH.exists():
        print("\nEvaluating on MADAR-6 dev...")
        dev = load_tsv(DEV_PATH)
        X_dev = featurize([t for t, _ in dev], markers, patterns)
        y_dev = np.array([CLASSES.index(c) for _, c in dev])
        y_pred = clf.predict(X_dev)
        dev_acc = accuracy_score(y_dev, y_pred)
        dev_f1 = f1_score(y_dev, y_pred, average="macro")
        print(f"  dev accuracy: {dev_acc:.4f}")
        print(f"  dev macro-F1: {dev_f1:.4f}")
        print()
        print(classification_report(y_dev, y_pred, target_names=CLASSES, digits=4))

    # Extract weights
    coef = clf.coef_         # [n_classes, n_features]
    intercept = clf.intercept_  # [n_classes]
    profile = {
        "version": 1,
        "source": "MADAR-Corpus-6-train (NYU AD shared task 2019)",
        "classes": CLASSES,
        "markers": markers,
        "coef": coef.astype(float).tolist(),
        "intercept": intercept.astype(float).tolist(),
        "regularization": "L2 (C=1.0)",
        "feature_clip": 3,
        "model": "sklearn.linear_model.LogisticRegression (multinomial, lbfgs)",
        "notes": (
            "Per-class linear weights over binary/count marker features. "
            "Inference is pure-Python: pre-normalize text, count marker hits "
            "(clipped at 3), compute scores = coef @ x + intercept, predict argmax. "
            "No sklearn or numpy required at inference time."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved learned profile to {OUTPUT_PATH}")
    print(f"  {len(markers)} markers × {len(CLASSES)} classes "
          f"({coef.size} weights + {len(CLASSES)} intercepts)")
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"  JSON size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
