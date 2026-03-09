"""
train.py
--------
Trains the EfficientNetV2 butterfly classifier.

Usage:
    python train.py

Outputs:
    model_weights.h5    -- best model weights (saved during training)
    label_classes.npy   -- label encoder classes (needed by predict.py)

Data expected at:
    data/train/             -- training images
    data/Training_set.csv   -- columns: filename, label
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from baseline.model import build_model, get_strategy, IMAGE_SIZE

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_DIR    = "data/train"
METADATA_CSV = "data/Training_set.csv"
WEIGHTS_FILE = "model_weights.h5"
CLASSES_FILE = "label_classes.npy"

EPOCHS           = 5
STEPS_PER_EPOCH  = 2000
BATCH_SIZE       = 64
TRAIN_SPLIT      = 0.8

# ── 1. Hardware strategy ───────────────────────────────────────────────────────
print("=" * 60)
print("🦋  Butterfly Classification — Training")
print("=" * 60)
strategy = get_strategy()

# ── 2. Metadata ────────────────────────────────────────────────────────────────
print("\nLoading metadata...")
meta_df = pd.read_csv(METADATA_CSV)
print(f"  Samples : {len(meta_df)}")
print(f"  Classes : {meta_df['label'].nunique()}")

# ── 3. Encode labels ───────────────────────────────────────────────────────────
label_encoder = LabelEncoder()
meta_df["label_encoding"] = label_encoder.fit_transform(meta_df["label"])
NUM_CLASSES = len(label_encoder.classes_)

# Save so predict.py can decode
np.save(CLASSES_FILE, label_encoder.classes_)
print(f"  Label classes saved to {CLASSES_FILE}")

# ── 4. Load images ─────────────────────────────────────────────────────────────
print(f"\nLoading and resizing images to {IMAGE_SIZE}x{IMAGE_SIZE}...")
images, label_encodings, failed = [], [], 0

for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    path = os.path.join(TRAIN_DIR, row["filename"])
    try:
        img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(img))
        label_encodings.append(row["label_encoding"])
    except Exception:
        failed += 1

if failed:
    print(f"  Warning: {failed} images failed to load and were skipped.")
print(f"  Loaded {len(images)} images successfully.")

# ── 5. Arrays ──────────────────────────────────────────────────────────────────
X = np.asarray(images, dtype="float32") / 255.0
y = np.asarray(label_encodings, dtype="float32")

# ── 6. Split ───────────────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=TRAIN_SPLIT, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

# ── 7. tf.data ─────────────────────────────────────────────────────────────────
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

# ── 8. Model ───────────────────────────────────────────────────────────────────
print("\nBuilding model...")
model = build_model(strategy, num_classes=NUM_CLASSES)
model.summary()

# ── 9. Train ───────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        WEIGHTS_FILE, save_best_only=True,
        monitor="val_sparse_categorical_accuracy",
        mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=3, mode="max", restore_best_weights=True, verbose=1
    ),
]

print(f"\nTraining for up to {EPOCHS} epochs...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    callbacks=callbacks,
)

# ── 10. Result ─────────────────────────────────────────────────────────────────
best = max(history.history["val_sparse_categorical_accuracy"])
print(f"\n{'='*60}")
print(f"Training complete!")
print(f"  Best validation accuracy : {best:.4f} ({best*100:.2f}%)")
print(f"  Weights saved to         : {WEIGHTS_FILE}")
print(f"\nNext step: python predict.py")
