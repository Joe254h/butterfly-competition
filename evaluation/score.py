import numpy as np
import json
import sys
import os
from sklearn.metrics import accuracy_score, f1_score

if len(sys.argv) < 2:
    print("Usage: python evaluation/score.py <submission_file>")
    sys.exit(1)

submission_file = sys.argv[1]
labels_file     = "evaluation/test_labels.csv"

if not os.path.exists(labels_file):
    print("Test labels not found.")
    sys.exit(1)

if not os.path.exists(submission_file):
    print(f"Submission not found: {submission_file}")
    sys.exit(1)

if os.path.getsize(submission_file) == 0:
    print("Submission file is empty.")
    sys.exit(1)

# Read labels
y_true = open(labels_file, 'r', encoding='utf-8-sig').read().strip().splitlines()
y_true = [s.strip().upper() for s in y_true if s.strip()]

# Read submission
y_pred = open(submission_file, 'r', encoding='utf-8-sig').read().strip().splitlines()
y_pred = [s.strip().upper() for s in y_pred if s.strip()]

print(f"Test labels : {len(y_true)}")
print(f"Predictions : {len(y_pred)}")
print(f"Sample pred : {y_pred[:3]}")

if len(y_pred) == 0:
    print("Submission file has no predictions.")
    sys.exit(1)

if len(y_pred) != len(y_true):
    print(f"Length mismatch: got {len(y_pred)}, expected {len(y_true)}")
    sys.exit(1)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = float(accuracy_score(y_true, y_pred))
f1       = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 (macro): {f1:.4f} ({f1*100:.2f}%)")

with open("score.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1_macro": f1}, f, indent=2)

print("score.json saved")
