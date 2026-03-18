import os
from pathlib import Path
from PIL import Image
import sys

# ---------------- CONFIG ----------------
DATA_ROOT = Path("D:/elysa/doctora-project/data")
SPLITS = ["train", "valid", "test"]

# Extensions supportées
IMG_EXT = [".jpg", ".jpeg", ".png"]

# ---------------- FONCTIONS ----------------

def get_long_path(path):
    """Prépare le chemin pour Windows pour supporter les noms longs (> 260 caractères)."""
    abs_path = str(path.absolute())
    if sys.platform == "win32" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

def is_image_valid(path):
    try:
        long_path = get_long_path(path)
        with Image.open(long_path) as img:
            img.verify()  # vérifie si image corrompue
        return True
    except:
        return False

def clean_dataset(img_dir, label_dir, split_name):
    print(f"\n--- NETTOYAGE DU SPLIT : {split_name.upper()} ---")

    # S'assurer que les dossiers existent
    if not img_dir.exists() or not label_dir.exists():
        print(f"Passage : Un des dossiers n'existe pas dans {split_name} (attendu: {img_dir} et {label_dir})")
        return

    # Listage initial
    imgs = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT}
    labels = {f.stem: f for f in label_dir.iterdir() if f.suffix == ".txt"}

    # Stats
    removed_imgs = 0
    removed_labels = 0
    corrupted_imgs = 0
    invalid_labels = 0

    # 1. Supprimer labels sans image
    for name, label_path in list(labels.items()):
        if name not in imgs:
            try:
                os.remove(get_long_path(label_path))
                removed_labels += 1
                del labels[name]
            except FileNotFoundError:
                pass

    # 2. Supprimer images sans label
    for name, img_path in list(imgs.items()):
        if name not in labels:
            try:
                os.remove(get_long_path(img_path))
                removed_imgs += 1
                del imgs[name]
            except FileNotFoundError:
                pass

    # 3. Vérifier images corrompues & 4. Vérifier labels YOLO
    for name, img_path in list(imgs.items()):
        label_path = labels.get(name)

        # Vérification image
        if not is_image_valid(img_path):
            try:
                os.remove(get_long_path(img_path))
                corrupted_imgs += 1
                if label_path and label_path.exists():
                    os.remove(get_long_path(label_path))
                    removed_labels += 1
                continue
            except FileNotFoundError:
                continue

        # Vérification label YOLO
        if label_path:
            is_valid = True
            try:
                with open(get_long_path(label_path), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            is_valid = False
                            break
            except Exception:
                is_valid = False

            if not is_valid:
                try:
                    os.remove(get_long_path(label_path))
                    invalid_labels += 1
                    if img_path.exists():
                        os.remove(get_long_path(img_path))
                        removed_imgs += 1
                except FileNotFoundError:
                    pass

    # ---------------- RÉSULTAT ----------------
    print(f"Images supprimées (orphelines/invalides): {removed_imgs}")
    print(f"Labels supprimés (orphelins/corrompus): {removed_labels}")
    print(f"Images corrompues supprimées: {corrupted_imgs}")
    print(f"Labels YOLO invalides supprimés: {invalid_labels}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    for split in SPLITS:
        img_dir = DATA_ROOT / split / "images"
        label_dir = DATA_ROOT / split / "labels"
        clean_dataset(img_dir, label_dir, split)
    print("\n=== TOUT LE DATASET EST NETTOYÉ ===")