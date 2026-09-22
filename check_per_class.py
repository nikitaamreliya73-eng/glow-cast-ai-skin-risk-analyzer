import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model/best_skin_type_cnn_v5.keras"
TEST_DIR = "skin_dataset/test"

print("\n======================================")
print("GLOW CAST V5 PER-CLASS PERFORMANCE")
print("======================================")

model = tf.keras.models.load_model(MODEL_PATH)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

class_names = test_ds.class_names

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

cm = confusion_matrix(y_true, y_pred)

print("Total test images:", len(y_true))

print("\nCLASS PERFORMANCE")
print("--------------------------------------")

for name in class_names:
    print(
        f"{name.upper():8} | "
        f"Precision: {report[name]['precision'] * 100:.2f}% | "
        f"Recall: {report[name]['recall'] * 100:.2f}% | "
        f"F1: {report[name]['f1-score'] * 100:.2f}% | "
        f"Support: {int(report[name]['support'])}"
    )

print("\nOVERALL")
print("--------------------------------------")

print(
    f"Accuracy: {report['accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1: {report['macro avg']['f1-score'] * 100:.2f}%"
)

print(
    f"Weighted F1: {report['weighted avg']['f1-score'] * 100:.2f}%"
)

print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted")
print("--------------------------------------")
print("             DRY  NORMAL  OILY")

for i, name in enumerate(class_names):
    print(
        f"{name.upper():8} "
        f"{cm[i].tolist()}"
    )

print("======================================")