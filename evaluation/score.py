import numpy as np
import pandas as pd
import json
import sys
import os
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
    classification_report
)

if len(sys.argv) < 2:
    print("Usage: python evaluation/score.py <submission_file>")
    sys.exit(1)

submission_file = sys.argv[1]
labels_file     = "evaluation/test_labels.csv"

if not os.path.exists(labels_file):
    print(f"Test labels not found: {labels_file}")
    sys.exit(1)

y_true = pd.read_csv(labels_file, header=None).iloc[:, 0].values.astype(str)

if not os.path.exists(submission_file):
    print(f"Submission not found: {submission_file}")
    sys.exit(1)

try:
    y_pred = pd.read_csv(submission_file, header=None).iloc[:, 0].values.astype(str)
except Exception as e:
    print(f"Could not read submission: {e}")
    sys.exit(1)

if len(y_pred) != len(y_true):
    print(f"Length mismatch: got {len(y_pred)}, expected {len(y_true)}")
    sys.exit(1)

y_pred = np.array([s.strip().upper() for s in y_pred])
y_true = np.array([s.strip().upper() for s in y_true])

accuracy  = float(accuracy_score(y_true, y_pred))
f1        = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
recall    = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

print(f"Accuracy         : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"F1 (macro)       : {f1:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro)   : {recall:.4f}")

report   = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
class_f1 = {k: v['f1-score'] for k, v in report.items()
            if k not in ['accuracy', 'macro avg', 'weighted avg']}
worst = sorted(class_f1.items(), key=lambda x: x[1])[:5]
print("5 hardest species:")
for species, val in worst:
    print(f"  {species:<35} F1={val:.3f}")

with open("score.json", "w") as f:
    json.dump({
        "accuracy":  accuracy,
        "f1_macro":  f1,
        "precision": precision,
        "recall":    recall,
    }, f, indent=2)

print("score.json saved")
