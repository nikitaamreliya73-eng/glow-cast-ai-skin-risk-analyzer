import tensorflow as tf
import numpy as np
import os

from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# GLOW CAST — CNN V3 EVALUATION
# ==========================================

TEST_DIR = "skin_dataset/test"
MODEL_PATH = "model/best_skin_type_cnn_v5.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("\n==============================")
print("GLOW CAST CNN V4 EVALUATION")
print("==============================")


# ==========================================
# LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...")

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
# LOAD MODEL
# ==========================================

print("\nLoading best CNN V3 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==========================================
# TEST EVALUATION
# ==========================================

print("\n==============================")
print("TEST EVALUATION")
print("==============================")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)


# ==========================================
# GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

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


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ==========================================
# COLLECT TEST IMAGE PATHS
# ==========================================

print("\n==============================")
print("COLLECTING TEST IMAGE PATHS")
print("==============================")

image_paths = []

for class_index, class_name in enumerate(
    class_names
):

    class_folder = os.path.join(
        TEST_DIR,
        class_name
    )

    for filename in sorted(
        os.listdir(class_folder)
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        ):

            image_paths.append(
                (
                    os.path.join(
                        class_folder,
                        filename
                    ),
                    class_index
                )
            )


print(
    "Total test images found:",
    len(image_paths)
)


# ==========================================
# DETAILED PREDICTION ANALYSIS
# ==========================================

print("\n==============================")
print("DETAILED PREDICTION ANALYSIS")
print("==============================")

correct_count = 0
wrong_count = 0


for image_path, actual_index in image_paths:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(
        prediction
    )

    confidence = (
        prediction[predicted_index]
        * 100
    )

    actual_class = class_names[
        actual_index
    ]

    predicted_class = class_names[
        predicted_index
    ]

    if actual_index == predicted_index:

        correct_count += 1

    else:

        wrong_count += 1


    print("\n----------------------------------------")

    print(
        "Image:",
        os.path.basename(image_path)
    )

    print(
        "Actual:",
        actual_class.upper()
    )

    print(
        "Predicted:",
        predicted_class.upper()
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print(
        "Probabilities:"
    )

    for class_name, probability in zip(
        class_names,
        prediction
    ):

        print(
            f"  {class_name.upper():8}: "
            f"{probability * 100:.2f}%"
        )


# ==========================================
# PREDICTION SUMMARY
# ==========================================

print("\n\n==============================")
print("PREDICTION SUMMARY")
print("==============================")

print(
    "\nTotal images:",
    len(image_paths)
)

print(
    "Correct predictions:",
    correct_count
)

print(
    "Wrong predictions:",
    wrong_count
)


if len(image_paths) > 0:

    manual_accuracy = (
        correct_count
        / len(image_paths)
    ) * 100

    print(
        "Manual Accuracy:",
        f"{manual_accuracy:.2f}%"
    )


# ==========================================
# PREDICTED CLASS COUNTS
# ==========================================

print("\n==============================")
print("PREDICTED CLASS COUNTS")
print("==============================")

for class_index, class_name in enumerate(
    class_names
):

    count = np.sum(
        y_pred == class_index
    )

    print(
        f"{class_name.upper():8}: {count}"
    )


# ==========================================
# CLASS-WISE PROBABILITY ANALYSIS
# ==========================================

print("\n==============================")
print("CLASS-WISE PROBABILITY ANALYSIS")
print("==============================")


class_probabilities = {
    class_name: []
    for class_name in class_names
}


# Store prediction probabilities
# according to the actual class

for image_path, actual_index in image_paths:

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    actual_class = class_names[
        actual_index
    ]

    class_probabilities[
        actual_class
    ].append(
        prediction
    )


# ==========================================
# AVERAGE PROBABILITIES
# ==========================================

for class_name in class_names:

    probabilities = np.array(
        class_probabilities[
            class_name
        ]
    )

    average_probability = probabilities.mean(
        axis=0
    )

    print("\n----------------------------------------")

    print(
        f"ACTUAL {class_name.upper()} IMAGES"
    )

    print("----------------------------------------")

    for predicted_class, probability in zip(
        class_names,
        average_probability
    ):

        print(
            f"{predicted_class.upper():8}: "
            f"{probability * 100:.2f}%"
        )


# ==========================================
# PROBABILITY GAP ANALYSIS
# ==========================================

print("\n==============================")
print("PROBABILITY GAP ANALYSIS")
print("==============================")


for class_name in class_names:

    probabilities = np.array(
        class_probabilities[
            class_name
        ]
    )

    average_probability = probabilities.mean(
        axis=0
    )

    actual_index = class_names.index(
        class_name
    )

    actual_probability = (
        average_probability[
            actual_index
        ]
    )

    other_probabilities = np.delete(
        average_probability,
        actual_index
    )

    highest_wrong_probability = np.max(
        other_probabilities
    )

    probability_gap = (
        actual_probability
        - highest_wrong_probability
    )


    print("\n----------------------------------------")

    print(
        f"ACTUAL {class_name.upper()}"
    )

    print(
        "Correct-class probability:",
        f"{actual_probability * 100:.2f}%"
    )

    print(
        "Highest wrong-class probability:",
        f"{highest_wrong_probability * 100:.2f}%"
    )

    print(
        "Probability gap:",
        f"{probability_gap * 100:.2f}%"
    )


# ==========================================
# FINAL RESULT
# ==========================================

print("\n==============================")
print("PROBABILITY ANALYSIS COMPLETED")
print("==============================")

print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")