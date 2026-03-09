import numpy as np
import pandas as pd
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

try:
    y_true = pd.read_csv(labels_file, header=None).iloc[:, 0].values.astype(str)
    y_pred = pd.read_csv(submission_file, header=None).iloc[:, 0].values.astype(str)
except Exception as e:
    print(f"Could not read files: {e}")
    sys.exit(1)

if len(y_pred) != len(y_true):
    print(f"Length mismatch: got {len(y_pred)}, expected {len(y_true)}")
    sys.exit(1)

y_true = np.array([s.strip().upper() for s in y_true])
y_pred = np.array([s.strip().upper() for s in y_pred])

accuracy = float(accuracy_score(y_true, y_pred))
f1       = float(f1_score(y_true, y_pred, average='macro', zero_division=0))

print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 (macro): {f1:.4f} ({f1*100:.2f}%)")

with open("score.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1_macro": f1}, f, indent=2)

print("score.json saved")
