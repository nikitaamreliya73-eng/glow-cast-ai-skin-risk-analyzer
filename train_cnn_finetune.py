import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# GLOW CAST — MOBILE NET V2 FINE-TUNING
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"


print("\n==============================")
print("GLOW CAST CNN FINE-TUNING")
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
# 3. LOAD PRETRAINED MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# ==========================================
# 4. UNFREEZE TOP LAYERS
# ==========================================

base_model.trainable = True


# Freeze most of MobileNetV2
# Only last 30 layers will be fine-tuned

for layer in base_model.layers[:-30]:

    layer.trainable = False


print("\nTrainable MobileNetV2 layers:")

for layer in base_model.layers:

    if layer.trainable:

        print(layer.name)


# ==========================================
# 5. BUILD MODEL
# ==========================================

model = models.Sequential([

    layers.Input(
        shape=(224, 224, 3)
    ),

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
# 6. COMPILE
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# 7. MODEL SUMMARY
# ==========================================

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

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

    "model/best_skin_type_cnn_finetuned.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max"

)


# ==========================================
# 9. TRAIN
# ==========================================

print("\n==============================")
print("STARTING FINE-TUNING")
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
# 10. SAVE MODEL
# ==========================================

model.save(
    "model/skin_type_cnn_finetuned.keras"
)


# ==========================================
# 11. FINAL RESULT
# ==========================================

print("\n==============================")
print("FINE-TUNING COMPLETED")
print("==============================")

print("\nClasses:")
print(class_names)

print("\nBest fine-tuned model:")
print(
    "model/best_skin_type_cnn_finetuned.keras"
)

print("\nFinal fine-tuned model:")
print(
    "model/skin_type_cnn_finetuned.keras"
)

print("\nGLOW CAST fine-tuned CNN ready!")