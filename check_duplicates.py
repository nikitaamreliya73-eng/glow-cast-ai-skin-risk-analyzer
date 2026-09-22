import os
import hashlib


# ==========================================
# GLOW CAST — DUPLICATE IMAGE CHECK
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
print("GLOW CAST DUPLICATE CHECK")
print("==============================")


# ==========================================
# STORE IMAGE HASHES
# ==========================================

hashes = {}

total_images = 0
duplicate_images = 0


for split in SPLITS:

    print(f"\nScanning {split.upper()}...")

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_DIR,
            split,
            class_name
        )

        if not os.path.exists(folder):
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

                with open(
                    image_path,
                    "rb"
                ) as file:

                    image_data = file.read()

                image_hash = hashlib.md5(
                    image_data
                ).hexdigest()


                total_images += 1


                if image_hash in hashes:

                    duplicate_images += 1

                    print("\nDUPLICATE FOUND")

                    print(
                        "Original:",
                        hashes[image_hash]
                    )

                    print(
                        "Duplicate:",
                        image_path
                    )

                else:

                    hashes[image_hash] = image_path


            except Exception:

                pass


# ==========================================
# FINAL RESULT
# ==========================================

print("\n==============================")
print("DUPLICATE CHECK RESULT")
print("==============================")

print(
    "Total images:",
    total_images
)

print(
    "Unique images:",
    len(hashes)
)

print(
    "Duplicate images:",
    duplicate_images
)


if duplicate_images == 0:

    print(
        "\nNo exact duplicate images found."
    )

else:

    print(
        "\nDuplicate images were found."
    )


print("\n==============================")
print("CHECK COMPLETED")
print("==============================")