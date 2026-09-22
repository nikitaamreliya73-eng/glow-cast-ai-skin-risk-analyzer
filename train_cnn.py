import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# GLOW CAST — IMPROVED CNN SKIN CLASSIFIER
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"


print("\n==============================")
print("GLOW CAST IMPROVED CNN")
print("==============================")


# ==========================================
# 1. LOAD DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
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
# 2. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),

    layers.RandomContrast(0.1)

])


# ==========================================
# 3. PRETRAINED MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# Freeze pretrained layers

base_model.trainable = False


# ==========================================
# 4. BUILD MODEL
# ==========================================

model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.4),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.3),

    layers.Dense(
        3,
        activation="softmax"
    )

])


# ==========================================
# 5. COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# 6. MODEL SUMMARY
# ==========================================

print("\nModel Summary:")

model.summary()


# ==========================================
# 7. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


checkpoint = ModelCheckpoint(

    "model/best_skin_type_cnn.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)


# ==========================================
# 8. TRAIN
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
# 9. SAVE FINAL MODEL
# ==========================================

model.save(
    "model/skin_type_cnn_improved.keras"
)


# ==========================================
# 10. FINAL RESULT
# ==========================================

print("\n==============================")
print("TRAINING COMPLETED")
print("==============================")

print("\nClasses:")
print(class_names)

print("\nBest model:")
print("model/best_skin_type_cnn.keras")

print("\nFinal model:")
print("model/skin_type_cnn_improved.keras")

print("\nGLOW CAST improved CNN ready!")