import os
from PIL import Image


# ==========================================
# GLOW CAST — DATASET QUALITY CHECK
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
print("GLOW CAST DATASET QUALITY CHECK")
print("==============================")


total_images = 0
total_valid = 0
total_corrupt = 0


for split in SPLITS:

    print(f"\n{split.upper()}")
    print("------------------------------")

    split_total = 0
    split_valid = 0
    split_corrupt = 0

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        class_total = 0
        class_valid = 0
        class_corrupt = 0

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

            class_total += 1

            image_path = os.path.join(
                folder,
                filename
            )

            try:

                with Image.open(image_path) as image:

                    image.verify()

                class_valid += 1

            except Exception:

                class_corrupt += 1

                print(
                    "\nCORRUPT IMAGE:"
                )

                print(
                    image_path
                )

        print(
            f"{class_name.upper():8} : "
            f"{class_total} images | "
            f"{class_valid} valid | "
            f"{class_corrupt} corrupt"
        )

        split_total += class_total
        split_valid += class_valid
        split_corrupt += class_corrupt

    print("------------------------------")

    print(
        f"TOTAL    : {split_total}"
    )

    print(
        f"VALID    : {split_valid}"
    )

    print(
        f"CORRUPT  : {split_corrupt}"
    )

    total_images += split_total
    total_valid += split_valid
    total_corrupt += split_corrupt


print("\n==============================")
print("FINAL DATASET CHECK")
print("==============================")

print(
    "Total images:",
    total_images
)

print(
    "Valid images:",
    total_valid
)

print(
    "Corrupt images:",
    total_corrupt
)

print("\n==============================")
print("CHECK COMPLETED")
print("==============================")