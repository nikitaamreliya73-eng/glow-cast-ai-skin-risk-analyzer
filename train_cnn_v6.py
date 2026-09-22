import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import numpy as np

# ============================================================
# GLOW CAST — CNN V6
# Class-Weighted Training Experiment
# ============================================================

DATASET_DIR = Path("skin_dataset")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

MODEL_PATH = "model/best_skin_type_cnn_v6.keras"

# ============================================================
# LOAD DATASET
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

class_names = train_ds.class_names

print()
print("CLASS NAMES:")
print(class_names)

# ============================================================
# DATASET PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print()
print("TRAIN DATASET READY")
print("VALIDATION DATASET READY")

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = {
    "dry": 652,
    "normal": 1073,
    "oily": 1000
}

total_samples = sum(class_counts.values())
num_classes = len(class_counts)

class_weight = {}

for index, class_name in enumerate(class_names):
    class_weight[index] = (
        total_samples /
        (num_classes * class_counts[class_name])
    )

print()
print("CLASS WEIGHTS:")

for index, class_name in enumerate(class_names):
    print(
        f"{class_name.upper():8s}: "
        f"{class_weight[index]:.4f}"
    )

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.03),
    ],
    name="data_augmentation"
)

# ============================================================
# MOBILE NET V2
# ============================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# ============================================================
# MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.35
)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(
    inputs,
    outputs
)

# ============================================================
# STAGE 1
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    ),

    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )
]

print()
print("=" * 42)
print("STAGE 1 — CLASS-WEIGHTED TRAINING")
print("=" * 42)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    class_weight=class_weight,
    callbacks=callbacks
)

# ============================================================
# STAGE 2 — FINE TUNING
# ============================================================

print()
print("=" * 42)
print("STAGE 2 — CLASS-WEIGHTED FINE-TUNING")
print("=" * 42)

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight,
    callbacks=callbacks
)

# ============================================================
# FINAL SAVE
# ============================================================

model.save(MODEL_PATH)

print()
print("=" * 42)
print("V6 TRAINING COMPLETE")
print("=" * 42)

print("Model saved:")
print(MODEL_PATH)

print()
print("Classes:")
print(class_names)

print()
print("V5 MODEL WAS NOT CHANGED")