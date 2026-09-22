import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==========================================
# GLOW CAST — CNN V5
# Two-Stage MobileNetV2 Fine-Tuning
# ==========================================

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

STAGE_1_EPOCHS = 12
STAGE_2_EPOCHS = 15

MODEL_PATH = "model/best_skin_type_cnn_v5.keras"


print("\n======================================")
print("GLOW CAST CNN V5 TRAINING")
print("======================================")


# ==========================================
# LOAD DATASET
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

class_names = train_ds.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# PREFETCH
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
validation_ds = validation_ds.prefetch(AUTOTUNE)


# ==========================================
# DATA AUGMENTATION
# ==========================================

print("\nCreating data augmentation...")

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.03)
], name="data_augmentation")


# ==========================================
# LOAD MOBILENETV2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


# ==========================================
# BUILD MODEL
# ==========================================

print("\nBuilding CNN V5...")

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

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
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.35
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
# STAGE 1 — FROZEN BACKBONE
# ==========================================

print("\n======================================")
print("STAGE 1 — FROZEN MOBILENETV2")
print("======================================")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


checkpoint_stage1 = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stage1 = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

reduce_stage1 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.000001,
    verbose=1
)

print("\nStarting Stage 1...\n")

history_stage1 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=STAGE_1_EPOCHS,
    callbacks=[
        checkpoint_stage1,
        early_stage1,
        reduce_stage1
    ]
)


# ==========================================
# STAGE 2 — CONTROLLED FINE-TUNING
# ==========================================

print("\n======================================")
print("STAGE 2 — CONTROLLED FINE-TUNING")
print("======================================")

base_model.trainable = True


# Freeze most of MobileNetV2
# Only last 20 layers will train

for layer in base_model.layers[:-20]:
    layer.trainable = False

print("\nTrainable MobileNetV2 layers:")

trainable_count = 0

for layer in base_model.layers:

    if layer.trainable:
        trainable_count += 1
        print(layer.name)

print(
    "\nTrainable backbone layers:",
    trainable_count
)


# ==========================================
# LOW LEARNING RATE
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint_stage2 = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stage2 = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_stage2 = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.0000001,
    verbose=1
)

print("\nStarting Stage 2...\n")

history_stage2 = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=STAGE_2_EPOCHS,
    callbacks=[
        checkpoint_stage2,
        early_stage2,
        reduce_stage2
    ]
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

final_path = "model/skin_type_cnn_v5.keras"

model.save(
    final_path
)


# ==========================================
# FINAL RESULTS
# ==========================================

best_stage1 = max(
    history_stage1.history["val_accuracy"]
)

best_stage2 = max(
    history_stage2.history["val_accuracy"]
)

best_validation_accuracy = max(
    best_stage1,
    best_stage2
)

print("\n======================================")
print("CNN V5 TRAINING COMPLETED")
print("======================================")

print("\nBest model saved:")
print(MODEL_PATH)

print("\nFinal model saved:")
print(final_path)

print("\n======================================")
print("BEST VALIDATION RESULTS")
print("======================================")

print(
    "Stage 1 Best Validation Accuracy:",
    f"{best_stage1 * 100:.2f}%"
)

print(
    "Stage 2 Best Validation Accuracy:",
    f"{best_stage2 * 100:.2f}%"
)

print(
    "Overall Best Validation Accuracy:",
    f"{best_validation_accuracy * 100:.2f}%"
)

print("\n======================================")
print("DONE")
print("======================================")