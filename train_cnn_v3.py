import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


# ==========================================
# GLOW CAST — CNN V3
# Class Weighted MobileNetV2
# ==========================================

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

MODEL_PATH = "model/best_skin_type_cnn_v3.keras"


print("\n==============================")
print("GLOW CAST CNN V3 TRAINING")
print("==============================")


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)


print("\nLoading validation dataset...")

validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 2. CLASS NAMES
# ==========================================

class_names = train_ds.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# 3. CALCULATE CLASS WEIGHTS
# ==========================================

print("\nCalculating class weights...")


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


print("\nClass counts:")

for class_name, count in zip(
    class_names,
    class_counts
):
    print(
        f"{class_name.upper():8} : {count}"
    )


classes = np.arange(
    len(class_names)
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.concatenate([
        np.full(
            count,
            class_index
        )
        for class_index, count
        in enumerate(class_counts)
    ])
)


class_weights = {
    class_index: float(weight)
    for class_index, weight
    in zip(
        classes,
        class_weights_array
    )
}


print("\nClass weights:")

for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index].upper():8} : {weight:.4f}"
    )


# ==========================================
# 4. DATA PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    AUTOTUNE
)


# ==========================================
# 5. DATA AUGMENTATION
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
# 6. PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers

base_model.trainable = False


# ==========================================
# 7. BUILD MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)


# Data augmentation

x = data_augmentation(
    inputs
)


# MobileNetV2 preprocessing

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    x
)


# Feature extraction

x = base_model(
    x,
    training=False
)


# Global pooling

x = layers.GlobalAveragePooling2D()(
    x
)


# Batch normalization

x = layers.BatchNormalization()(
    x
)


# Dense layer

x = layers.Dense(
    256,
    activation="relu"
)(x)


# Dropout

x = layers.Dropout(
    0.5
)(x)


# Second dense layer

x = layers.Dense(
    128,
    activation="relu"
)(x)


# Dropout

x = layers.Dropout(
    0.3
)(x)


# Output

outputs = layers.Dense(
    3,
    activation="softmax"
)(x)


# Create model

model = models.Model(
    inputs,
    outputs
)


# ==========================================
# 8. COMPILE MODEL
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
# 9. MODEL SUMMARY
# ==========================================

print("\n==============================")
print("MODEL SUMMARY")
print("==============================\n")

model.summary()


# ==========================================
# 10. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)


# ==========================================
# 11. START TRAINING
# ==========================================

print("\n==============================")
print("STARTING CNN V3 TRAINING")
print("==============================\n")


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ==========================================
# 12. SAVE FINAL MODEL
# ==========================================

final_path = "model/skin_type_cnn_v3.keras"

model.save(
    final_path
)


# ==========================================
# 13. TRAINING COMPLETED
# ==========================================

print("\n==============================")
print("CNN V3 TRAINING COMPLETED")
print("==============================")


print("\nBest model saved:")

print(
    MODEL_PATH
)


print("\nFinal model saved:")

print(
    final_path
)


print("\n==============================")
print("CLASS WEIGHTS USED")
print("==============================")


for class_index, weight in class_weights.items():

    print(
        f"{class_names[class_index].upper():8} : {weight:.4f}"
    )


print("\n==============================")
print("DONE")
print("==============================")