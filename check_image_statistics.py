from PIL import Image
import os
import numpy as np

# ==========================================
# GLOW CAST — IMAGE STATISTICS CHECK
# ==========================================

DATASET_DIR = "skin_dataset"

SPLITS = ["train", "validation", "test"]
CLASSES = ["dry", "normal", "oily"]

print("\n======================================")
print("GLOW CAST IMAGE STATISTICS CHECK")
print("======================================")

for split in SPLITS:

    print(f"\n\n{split.upper()}")
    print("-" * 50)

    for class_name in CLASSES:

        folder = os.path.join(DATASET_DIR, split, class_name)

        brightness_values = []
        contrast_values = []

        image_count = 0

        for filename in os.listdir(folder):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            filepath = os.path.join(folder, filename)

            try:
                image = Image.open(filepath).convert("RGB")

                image_array = np.array(image, dtype=np.float32)

                # Average brightness
                brightness = image_array.mean()

                # Standard deviation = simple contrast measure
                contrast = image_array.std()

                brightness_values.append(brightness)
                contrast_values.append(contrast)

                image_count += 1

            except Exception:
                pass

        if image_count > 0:

            avg_brightness = np.mean(brightness_values)
            avg_contrast = np.mean(contrast_values)

            print(
                f"{class_name.upper():8} : "
                f"{image_count:4} images | "
                f"Brightness: {avg_brightness:6.2f} | "
                f"Contrast: {avg_contrast:6.2f}"
            )

print("\n======================================")
print("STATISTICS CHECK COMPLETED")
print("======================================")