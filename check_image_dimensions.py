import os
from PIL import Image
from collections import Counter


# ==========================================
# GLOW CAST — IMAGE DIMENSION CHECK
# ==========================================

DATASET_DIR = "skin_dataset"

SPLITS = [
    "train",
    "validation",
    "test"
]

CLASSES = [
    "dry",
    "normal",
    "oily"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)


print("\n==============================")
print("GLOW CAST IMAGE DIMENSION CHECK")
print("==============================")


all_dimensions = Counter()

total_images = 0


for split in SPLITS:

    print(f"\n{split.upper()}")
    print("------------------------------")

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        dimensions = Counter()

        class_total = 0

        if not os.path.exists(folder):

            print(
                f"{class_name.upper():8} : FOLDER NOT FOUND"
            )

            continue

        for filename in os.listdir(folder):

            if not filename.lower().endswith(
                VALID_EXTENSIONS
            ):
                continue

            image_path = os.path.join(
                folder,
                filename
            )

            try:

                with Image.open(image_path) as image:

                    width, height = image.size

                dimensions[
                    (width, height)
                ] += 1

                all_dimensions[
                    (width, height)
                ] += 1

                class_total += 1
                total_images += 1

            except Exception:

                pass


        print(
            f"\n{class_name.upper()}"
        )

        print(
            f"Total images: {class_total}"
        )

        print(
            "Dimensions:"
        )

        for dimension, count in dimensions.most_common(10):

            print(
                f"  {dimension[0]} x {dimension[1]}"
                f"  →  {count} images"
            )


print("\n==============================")
print("OVERALL DIMENSION SUMMARY")
print("==============================")


print(
    "Total images:",
    total_images
)


print(
    "\nMost common image dimensions:"
)


for dimension, count in all_dimensions.most_common(20):

    print(
        f"{dimension[0]} x {dimension[1]}"
        f"  →  {count} images"
    )


print("\n==============================")
print("CHECK COMPLETED")
print("==============================")