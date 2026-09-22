import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# GLOW CAST — SKIN TYPE PREDICTION
# ==========================================

IMG_SIZE = (224, 224)

MODEL_PATH = "model/best_skin_type_cnn.keras"

IMAGE_PATH = "test_photo.jpg"


print("\n==============================")
print("GLOW CAST AI SKIN ANALYSIS")
print("==============================")


# ==========================================
# 1. LOAD MODEL
# ==========================================

print("\nLoading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ==========================================
# 2. LOAD IMAGE
# ==========================================

print("\nLoading photo...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = image.resize(
    IMG_SIZE
)


# ==========================================
# 3. CONVERT IMAGE
# ==========================================

image_array = np.array(image)

image_array = image_array / 255.0

image_array = np.expand_dims(
    image_array,
    axis=0
)


# ==========================================
# 4. PREDICTION
# ==========================================

prediction = model.predict(
    image_array,
    verbose=0
)


# ==========================================
# 5. GET RESULT
# ==========================================

class_names = [
    "dry",
    "normal",
    "oily"
]

predicted_index = np.argmax(
    prediction[0]
)

predicted_skin_type = class_names[
    predicted_index
]

confidence = (
    prediction[0][predicted_index] * 100
)


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\n==============================")
print("SKIN ANALYSIS RESULT")
print("==============================")

print(
    "Predicted Skin Type:",
    predicted_skin_type.upper()
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("\nPrediction probabilities:")

for skin, probability in zip(
    class_names,
    prediction[0]
):
    print(
        skin.upper(),
        ":",
        round(probability * 100, 2),
        "%"
    )

print("\n==============================")
print("ANALYSIS COMPLETED")
print("==============================")