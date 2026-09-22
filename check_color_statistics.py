from PIL import Image
import os
import numpy as np

# ==========================================
# GLOW CAST — COLOR STATISTICS CHECK
# ==========================================

DATASET_DIR = "skin_dataset"

SPLITS = ["train", "validation", "test"]
CLASSES = ["dry", "normal", "oily"]

print("\n======================================")
print("GLOW CAST COLOR STATISTICS CHECK")
print("======================================")

for split in SPLITS:

    print(f"\n\n{split.upper()}")
    print("-" * 70)

    for class_name in CLASSES:

        folder = os.path.join(DATASET_DIR, split, class_name)

        red_values = []
        green_values = []
        blue_values = []

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

                # Separate RGB channels
                red = image_array[:, :, 0].mean()
                green = image_array[:, :, 1].mean()
                blue = image_array[:, :, 2].mean()

                red_values.append(red)
                green_values.append(green)
                blue_values.append(blue)

                image_count += 1

            except Exception:
                pass

        if image_count > 0:

            avg_red = np.mean(red_values)
            avg_green = np.mean(green_values)
            avg_blue = np.mean(blue_values)

            print(
                f"{class_name.upper():8} : "
                f"{image_count:4} images | "
                f"R: {avg_red:6.2f} | "
                f"G: {avg_green:6.2f} | "
                f"B: {avg_blue:6.2f}"
            )

print("\n======================================")
print("COLOR STATISTICS CHECK COMPLETED")
print("======================================")