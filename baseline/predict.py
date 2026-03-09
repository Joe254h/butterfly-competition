"""
predict.py
----------
Loads the trained model and generates predictions on the test set.

Usage:
    python predict.py

Requirements:
    model_weights.h5    -- trained model (run train.py first)
    label_classes.npy   -- label encoder classes (saved by train.py)
    data/test/          -- folder of test images

Output:
    submissions/submission.csv   -- one predicted label per row, no header

Then rename and submit:
    mv submissions/submission.csv submissions/YOUR_NAME_submission.csv
    git add submissions/YOUR_NAME_submission.csv
    git commit -m "My submission"
    git push
    --> open a Pull Request on GitHub
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from baseline.model import IMAGE_SIZE

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_DIR     = "data/test"
WEIGHTS_FILE = "model_weights.h5"
CLASSES_FILE = "label_classes.npy"
OUTPUT_FILE  = "submissions/submission.csv"
BATCH_SIZE   = 32

# ── 1. Check files ─────────────────────────────────────────────────────────────
print("=" * 60)
print("🦋  Butterfly Classification — Predict")
print("=" * 60)

if not os.path.exists(WEIGHTS_FILE):
    raise FileNotFoundError(f"Model weights not found: {WEIGHTS_FILE}\nRun train.py first.")

if not os.path.exists(CLASSES_FILE):
    raise FileNotFoundError(f"Label classes not found: {CLASSES_FILE}\nRun train.py first.")

# ── 2. Load label classes ──────────────────────────────────────────────────────
label_classes = np.load(CLASSES_FILE, allow_pickle=True)
print(f"Loaded {len(label_classes)} classes.")

# ── 3. Load model ──────────────────────────────────────────────────────────────
print("Loading model weights...")
model = tf.keras.models.load_model(WEIGHTS_FILE)
print("Model loaded.")

# ── 4. Load test images in sorted order ───────────────────────────────────────
print(f"\nLoading test images from {TEST_DIR}...")
test_files = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
print(f"Found {len(test_files)} test images.")

images = []
valid_files = []
failed = 0

for fname in tqdm(test_files, desc="Loading test images"):
    path = os.path.join(TEST_DIR, fname)
    try:
        img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(img, dtype="float32") / 255.0)
        valid_files.append(fname)
    except Exception:
        failed += 1

if failed:
    print(f"Warning: {failed} images failed to load and were skipped.")

X_test = np.asarray(images)
print(f"Test array shape: {X_test.shape}")

# ── 5. Predict ─────────────────────────────────────────────────────────────────
print("\nGenerating predictions...")
test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE)
pred_encodings = model.predict(test_ds, verbose=1).argmax(axis=1)

# Decode back to label names
pred_labels = label_classes[pred_encodings]

# ── 6. Save submission ─────────────────────────────────────────────────────────
os.makedirs("submissions", exist_ok=True)
pd.Series(pred_labels).to_csv(OUTPUT_FILE, index=False, header=False)

print(f"\nPredictions saved to: {OUTPUT_FILE}")
print(f"Total predictions  : {len(pred_labels)}")
print(f"\nNext step:")
print(f"  mv {OUTPUT_FILE} submissions/YOUR_NAME_submission.csv")
print(f"  git add submissions/YOUR_NAME_submission.csv")
print(f"  git commit -m 'My butterfly submission'")
print(f"  git push  -->  then open a Pull Request on GitHub")
