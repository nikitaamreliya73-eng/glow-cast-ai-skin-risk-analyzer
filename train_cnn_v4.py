import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np


# ==========================================
# GLOW CAST — CNN V4
# Balanced MobileNetV2
# ==========================================

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

MODEL_PATH = "model/best_skin_type_cnn_v4.keras"


print("\n======================================")
print("GLOW CAST CNN V4 TRAINING")
print("======================================")


# ==========================================
# LOAD TRAIN DATASET
# ==========================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


# ==========================================
# LOAD VALIDATION DATASET
# ==========================================

print("\nLoading validation dataset...")

validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_ds.class_names


print("\nClasses:")
print(class_names)


# ==========================================
# CLASS COUNTS
# ==========================================

print("\n======================================")
print("CLASS COUNTS")
print("======================================")


class_counts = []


for class_name in class_names:

    class_path = f"{TRAIN_DIR}/{class_name}"

    count = len([
        file
        for file in tf.io.gfile.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    class_counts.append(count)

    print(
        f"{class_name.upper():8}: {count}"
    )


# ==========================================
# CUSTOM CLASS WEIGHTS
# ==========================================

print("\n======================================")
print("CUSTOM CLASS WEIGHTS")
print("======================================")


# Class order:
# 0 = dry
# 1 = normal
# 2 = oily

class_weights = {
    0: 1.30,
    1: 1.20,
    2: 0.75
}


for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index].upper():8}: "
        f"{weight:.4f}"
    )


# ==========================================
# PREFETCH
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)


# ==========================================
# DATA AUGMENTATION
# ==========================================

print("\nCreating data augmentation...")


data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.08
    ),

    layers.RandomContrast(
        0.05
    )

], name="data_augmentation")


# ==========================================
# MOBILE NET V2
# ==========================================

print("\nLoading MobileNetV2...")


base_model = MobileNetV2(
    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Initially frozen
base_model.trainable = False


# ==========================================
# BUILD MODEL
# ==========================================

print("\nBuilding CNN V4...")


inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)


x = data_augmentation(
    inputs
)


x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.BatchNormalization()(x)


x = layers.Dense(
    256,
    activation="relu"
)(x)


x = layers.Dropout(
    0.4
)(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.25
)(x)


outputs = layers.Dense(
    3,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ==========================================
# COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ==========================================
# MODEL SUMMARY
# ==========================================

print("\n======================================")
print("MODEL SUMMARY")
print("======================================")

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.000001,

    verbose=1
)


# ==========================================
# START TRAINING
# ==========================================

print("\n======================================")
print("STARTING CNN V4 TRAINING")
print("======================================\n")


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        checkpoint,
        reduce_lr
    ]
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

final_path = (
    "model/skin_type_cnn_v4.keras"
)


model.save(
    final_path
)


# ==========================================
# TRAINING COMPLETED
# ==========================================

print("\n======================================")
print("CNN V4 TRAINING COMPLETED")
print("======================================")


print("\nBest model saved:")

print(
    MODEL_PATH
)


print("\nFinal model saved:")

print(
    final_path
)


# ==========================================
# BEST VALIDATION RESULTS
# ==========================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)


best_val_loss = min(
    history.history["val_loss"]
)


print("\n======================================")
print("BEST VALIDATION RESULTS")
print("======================================")


print(
    "Best Validation Accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)


print(
    "Best Validation Loss:",
    f"{best_val_loss:.4f}"
)


print("\n======================================")
print("DONE")
print("======================================")