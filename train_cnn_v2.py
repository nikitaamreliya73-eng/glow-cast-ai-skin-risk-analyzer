import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os


# ==========================================
# GLOW CAST — CNN V2
# ==========================================

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

MODEL_PATH = "model/best_skin_type_cnn_v2.keras"


print("\n==============================")
print("GLOW CAST CNN V2 TRAINING")
print("==============================")


# ==========================================
# 1. LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

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
# 2. DATA PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)


# ==========================================
# 3. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    
    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    ),

    layers.RandomContrast(
        0.10
    ),

], name="data_augmentation")


# ==========================================
# 4. PRETRAINED MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers initially

base_model.trainable = False


# ==========================================
# 5. BUILD MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

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
    0.5
)(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(
    0.3
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
# 6. COMPILE
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
# 7. MODEL SUMMARY
# ==========================================

print("\nModel Summary:\n")

model.summary()


# ==========================================
# 8. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\n==============================")
print("STARTING TRAINING")
print("==============================\n")


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# 10. SAVE FINAL MODEL
# ==========================================

final_path = "model/skin_type_cnn_v2.keras"

model.save(
    final_path
)


print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")


print(
    "\nBest model saved:"
)

print(
    MODEL_PATH
)


print(
    "\nFinal model saved:"
)

print(
    final_path
)