import os
import glob
import shutil

# ------------------------ CONFIG ------------------------
train_img_dir = "D:/elysa/doctora-project/data/train/images"
train_label_dir = "D:/elysa/doctora-project/data/train/labels"

valid_img_dir = "D:/elysa/doctora-project/data/valid/images"
valid_label_dir = "D:/elysa/doctora-project/data/valid/labels"

# Nombre de classes à garder
num_classes_to_keep = 20

# Dossier backup
backup_folder = "D:/elysa/doctora-project/data/backup"
os.makedirs(backup_folder, exist_ok=True)

# Fonction pour backup labels et images
def backup_files(label_dir, img_dir):
    for f in glob.glob(os.path.join(label_dir, "*.txt")):
        shutil.copy(f, os.path.join(backup_folder, os.path.basename(f)))
    for f in glob.glob(os.path.join(img_dir, "*.jpg")):
        shutil.copy(f, os.path.join(backup_folder, os.path.basename(f)))

print("Backup initial des images et labels...")
backup_files(train_label_dir, train_img_dir)
backup_files(valid_label_dir, valid_img_dir)
print("Backup terminé ✅")

# ------------------------ Fonction pour filtrer les labels et supprimer images vides ------------------------
def filter_labels_and_remove_images(label_dir, img_dir, num_classes):
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    removed_images = 0
    for file in txt_files:
        new_lines = []
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                if cls < num_classes:
                    new_lines.append(" ".join(parts))
        # Réécrire le fichier
        with open(file, "w") as f:
            f.write("\n".join(new_lines))
        # Supprimer l'image si le fichier est vide
        if len(new_lines) == 0:
            img_file = os.path.join(img_dir, os.path.basename(file).replace(".txt", ".jpg"))
            if os.path.exists(img_file):
                os.remove(img_file)
                removed_images += 1
    print(f"{removed_images} images supprimées dans {img_dir} car elles n'avaient pas de classe valide.")

# ------------------------ Nettoyage ------------------------
print("Filtrage des labels et suppression des images vides...")
filter_labels_and_remove_images(train_label_dir, train_img_dir, num_classes_to_keep)
filter_labels_and_remove_images(valid_label_dir, valid_img_dir, num_classes_to_keep)
print(f"Nettoyage terminé ✅ Seules les {num_classes_to_keep} premières classes sont conservées.")