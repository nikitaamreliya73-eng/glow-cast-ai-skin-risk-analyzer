import tensorflow as tf


# ==========================================
# GLOW CAST - SKIN DATASET CHECK
# ==========================================

TRAIN_DIR = "skin_dataset/train"
VALIDATION_DIR = "skin_dataset/validation"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("\n==============================")
print("GLOW CAST SKIN DATASET")
print("==============================")


# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# Load validation dataset
validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("\nClasses:")
print(train_ds.class_names)

print("\nDataset loaded successfully!")