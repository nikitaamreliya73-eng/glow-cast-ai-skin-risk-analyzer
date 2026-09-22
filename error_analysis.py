import tensorflow as tf
import numpy as np
from collections import Counter

MODEL_PATH = "model/best_skin_type_cnn_v5.keras"
TEST_DIR = "skin_dataset/test"

print("\n======================================")
print("GLOW CAST V5 ERROR ANALYSIS")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = test_ds.class_names

y_true = []
y_pred = []
probabilities = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    probabilities.extend(preds)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
probabilities = np.array(probabilities)

wrong = y_true != y_pred

print("\nTOTAL TEST IMAGES:", len(y_true))
print("CORRECT PREDICTIONS:", np.sum(~wrong))
print("WRONG PREDICTIONS:", np.sum(wrong))

print("\nWRONG PREDICTION PAIRS")
print("--------------------------------------")

pairs = Counter(
    (class_names[a], class_names[p])
    for a, p in zip(y_true[wrong], y_pred[wrong])
)

for (actual, predicted), count in pairs.most_common():
    print(
        f"Actual {actual.upper():7} -> "
        f"Predicted {predicted.upper():7}: {count} images"
    )

print("\nWRONG PREDICTIONS BY ACTUAL CLASS")
print("--------------------------------------")

for i, name in enumerate(class_names):
    total_actual = np.sum(y_true == i)
    wrong_actual = np.sum((y_true == i) & wrong)

    print(
        f"{name.upper():8} | "
        f"Total: {total_actual:2} | "
        f"Wrong: {wrong_actual:2} | "
        f"Error Rate: {(wrong_actual / total_actual) * 100:.2f}%"
    )

print("\nWRONG PREDICTIONS BY PREDICTED CLASS")
print("--------------------------------------")

for i, name in enumerate(class_names):
    wrong_predicted = np.sum((y_pred == i) & wrong)

    print(
        f"{name.upper():8} | "
        f"Wrongly predicted: {wrong_predicted}"
    )

print("\n======================================")
print("TOP CONFIDENT WRONG PREDICTIONS")
print("======================================")

wrong_indices = np.where(wrong)[0]

wrong_confidences = np.max(
    probabilities[wrong_indices],
    axis=1
)

order = np.argsort(
    wrong_confidences
)[::-1]

for position in order[:10]:
    idx = wrong_indices[position]

    print(
        f"Image #{idx + 1:3} | "
        f"Actual: {class_names[y_true[idx]].upper():7} | "
        f"Predicted: {class_names[y_pred[idx]].upper():7} | "
        f"Confidence: {probabilities[idx].max() * 100:.2f}%"
    )

print("\n======================================")