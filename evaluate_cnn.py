import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ==========================================
# GLOW CAST — CNN TEST EVALUATION
# ==========================================

TEST_DIR = "skin_dataset/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "model/best_skin_type_cnn_v3.keras"
print("\n==============================")
print("GLOW CAST CNN TEST")
print("==============================")


# ==========================================
# 1. LOAD TEST DATASET
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = test_ds.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 3. EVALUATE MODEL
# ==========================================

print("\nEvaluating test dataset...\n")

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1
)


print("\n==============================")
print("TEST RESULT")
print("==============================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Test Loss:",
    round(loss, 4)
)


# ==========================================
# 4. PREDICTIONS
# ==========================================

y_true = []
y_pred = []


for images, labels in test_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


# ==========================================
# 5. CLASSIFICATION REPORT
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")