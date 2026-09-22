import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# ==========================================
# GLOW CAST — CLEAN DATASET CNN
# ==========================================

DATASET_DIR = Path("skin_dataset_clean")

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

MODEL_PATH = "model/best_skin_type_cnn_clean.keras"


# ==========================================
# LOAD DATASET
# ==========================================

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

print("\nCLASS NAMES:")
print(class_names)

print("\nTRAIN DATASET READY")
print("VALIDATION DATASET READY")


# ==========================================
# PERFORMANCE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.03),
    ],
    name="data_augmentation"
)


# ==========================================
# BASE MODEL
# ==========================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# MODEL
# ==========================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)


# ==========================================
# STAGE 1
# ==========================================

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


print("\n==========================================")
print("STAGE 1 TRAINING")
print("==========================================")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=callbacks
)


# ==========================================
# STAGE 2 — FINE TUNING
# ==========================================

print("\n==========================================")
print("STAGE 2 FINE-TUNING")
print("==========================================")

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
    callbacks=callbacks
)


# ==========================================
# FINAL SAVE
# ==========================================

model.save(MODEL_PATH)

print("\n==========================================")
print("CLEAN MODEL TRAINING COMPLETE")
print("==========================================")

print("Model saved:", MODEL_PATH)
print("Classes:", class_names)