import os
import random
import shutil
from PIL import Image
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
IMG_SIZE = 224
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

CLASS_MAP = {
    "Cat": "cats",
    "Dog": "dogs"
}

def download_dataset():
    print("Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print("Downloading dataset from Kaggle...")
    dataset = "bhavikjikadara/dog-and-cat-classification-dataset"
    api.dataset_download_files(dataset, path="data", unzip=True)

    # Find PetImages directory dynamically
    petimages_dir = None
    for root, dirs, _ in os.walk("data"):
        if "PetImages" in dirs:
            petimages_dir = os.path.join(root, "PetImages")
            break

    if petimages_dir is None:
        raise RuntimeError("PetImages directory not found after Kaggle download")

    print("Found PetImages at:", petimages_dir)

    os.makedirs(RAW_DIR, exist_ok=True)

    for src_cls, dst_cls in CLASS_MAP.items():
        src = os.path.join(petimages_dir, src_cls)
        dst = os.path.join(RAW_DIR, dst_cls)

        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing class folder: {src}")

        if not os.path.exists(dst):
            shutil.move(src, dst)

    print("Dataset moved to", RAW_DIR)

def prepare_dirs():
    for split in SPLITS:
        for cls in CLASS_MAP.values():
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

def resize_and_save(src, dst, augment=False):
    try:
        img = Image.open(src).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img.save(dst)
        
        # Apply augmentation only if flag is True (i.e., for the training set)
        if augment:
            img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            base, ext = os.path.splitext(dst)
            aug_dst = f"{base}_aug{ext}"
            img_flipped.save(aug_dst)
            
    except Exception:
        pass  # Skip corrupted images (important for PetImages dataset)

def preprocess():
    prepare_dirs()

    for cls in CLASS_MAP.values():
        cls_dir = os.path.join(RAW_DIR, cls)
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)
        n = len(images)

        train_end = int(SPLITS["train"] * n)
        val_end = train_end + int(SPLITS["val"] * n)

        split_map = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in split_map.items():
            for f in files:
                src = os.path.join(cls_dir, f)
                dst = os.path.join(PROCESSED_DIR, split, cls, f)
                
                # Pass augment=True only if the split is 'train'
                is_train_split = (split == "train")
                resize_and_save(src, dst, augment=is_train_split)

            print(f"{cls} -> {split}: {len(files)} images (plus augmentations if train)")

if __name__ == "__main__":
    if not os.path.exists(os.path.join(RAW_DIR, "cats")) or \
       not os.path.exists(os.path.join(RAW_DIR, "dogs")):
        download_dataset()
    else:
        print("Raw dataset already exists. Skipping download.")

    preprocess()
    print("Preprocessing complete!")
