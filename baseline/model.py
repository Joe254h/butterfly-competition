"""
baseline/model.py
-----------------
EfficientNetV2 baseline for butterfly classification.
Modify this (or create your own model) to improve your score!

Key ideas to try:
  - Set trainable=True to fine-tune the backbone (slower but more accurate)
  - Add more Dense layers or change their sizes
  - Add BatchNormalization layers
  - Try a different backbone entirely
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

IMAGE_SIZE    = 224
NUM_CLASSES   = 75
LEARNING_RATE = 5e-4

EFFICIENTNET_URL = (
    "https://www.kaggle.com/models/google/efficientnet-v2/"
    "frameworks/TensorFlow2/variations/"
    "imagenet21k-b3-feature-vector/versions/1"
)


def build_model(strategy, num_classes: int = NUM_CLASSES, trainable: bool = False):
    """
    Builds and compiles the EfficientNetV2 model.

    Parameters
    ----------
    strategy    : tf.distribute strategy (returned by get_strategy())
    num_classes : number of butterfly species (default 75)
    trainable   : fine-tune the backbone? default False (faster training)

    Returns
    -------
    compiled keras model
    """
    with strategy.scope():
        backbone = hub.KerasLayer(EFFICIENTNET_URL, trainable=trainable)

        model = keras.Sequential([
            backbone,
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax'),
        ])

        model.build([None, IMAGE_SIZE, IMAGE_SIZE, 3])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy'],
        )

    return model


def get_strategy():
    """Auto-detects TPU > GPU > CPU and returns the right distribution strategy."""
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"TPU detected: {tpu.master()}")
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        strategy = tf.distribute.get_strategy()
        device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
        print(f"Running on {device}")

    print(f"Replicas: {strategy.num_replicas_in_sync}")
    return strategy
