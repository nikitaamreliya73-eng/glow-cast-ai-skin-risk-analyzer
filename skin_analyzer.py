import tensorflow as tf
from PIL import Image
import numpy as np


# ==========================================
# GLOW CAST — TENSORFLOW SKIN ANALYZER
# ==========================================

IMG_SIZE = (224, 224)


def load_skin_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def analyze_skin(image_path, model_path):

    # Load TensorFlow model
    model = tf.keras.models.load_model(
        model_path
    )

    # Prepare image
    image = load_skin_image(
        image_path
    )

    # Prediction
    prediction = model.predict(
        image,
        verbose=0
    )

    return prediction


if __name__ == "__main__":

    print("\n==============================")
    print("GLOW CAST AI SKIN ANALYZER")
    print("==============================")

    print("TensorFlow Version:", tf.__version__)

    print("\nSkin analyzer module loaded successfully.")