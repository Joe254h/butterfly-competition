"""
evaluation/score.py
-------------------
Evaluates a butterfly submission against hidden test labels.

Called automatically by GitHub Actions -- participants do NOT run this.

Usage (GitHub Actions only):
    python evaluation/score.py submissions/someone_submission.csv

Output:
    score.json   -- contains accuracy score
"""

import pandas as pd
import json
import sys
import os

# ── 1. Parse args ──────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python evaluation/score.py <submission_file>")
    sys.exit(1)

submission_file = sys.argv[1]

# ── 2. Load submission ─────────────────────────────────────────────────────────
if not os.path.exists(submission_file):
    print(f"ERROR: Submission not found: {submission_file}")
    sys.exit(1)

try:
    y_pred = pd.read_csv(submission_file, header=None).iloc[:, 0].astype(str).values
except Exception as e:
    print(f"ERROR: Could not read submission: {e}")
    sys.exit(1)

# ── 3. Load hidden test labels ─────────────────────────────────────────────────
labels_file = "evaluation/test_labels.csv"
if not os.path.exists(labels_file):
    print(f"ERROR: Test labels not found: {labels_file}")
    sys.exit(1)

y_true = pd.read_csv(labels_file, header=None).iloc[:, 0].astype(str).values

# ── 4. Validate ────────────────────────────────────────────────────────────────
if len(y_pred) != len(y_true):
    print(f"ERROR: Submission has {len(y_pred)} predictions, expected {len(y_true)}.")
    sys.exit(1)

# Check labels are valid
valid_labels = set(y_true)
invalid = [p for p in y_pred if p not in valid_labels]
if invalid:
    print(f"WARNING: {len(invalid)} predictions contain unknown labels.")
    print(f"  Examples: {invalid[:5]}")

# ── 5. Compute accuracy ────────────────────────────────────────────────────────
accuracy = float((y_pred == y_true).mean())

print(f"Evaluation complete.")
print(f"  File       : {submission_file}")
print(f"  Samples    : {len(y_true)}")
print(f"  Accuracy   : {accuracy:.4f} ({accuracy*100:.2f}%)")

# ── 6. Save score ──────────────────────────────────────────────────────────────
with open("score.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print("  Score saved to score.json")
