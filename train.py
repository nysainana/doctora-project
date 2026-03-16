import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from model import YOLOv11nMobileNet

# --- Dataset YOLO ---
class YOLODataset(Dataset):
    """
    Dataset simple pour charger des images et des labels au format YOLO.
    Format Label YOLO : class x_center y_center width height (normalisé 0-1)
    """
    def __init__(self, img_dir, label_dir, img_size=320, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Charger le label correspondant
        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    # class x_c y_c w h
                    labels.append([float(x) for x in line.split()])
        
        # Pour simplifier l'exemple d'entraînement (Target Matching est complexe dans YOLO)
        # On retourne l'image et une cible factice ou simplifiée. 
        # Note : Un vrai entraînement nécessite un "Collate Function" car le nombre d'objets varie.
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels)

def collate_fn(batch):
    """Gère le fait que chaque image a un nombre de boîtes différent."""
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets

# --- Fonction de Loss Simplifiée ---
class SimpleYOLOLoss(nn.Module):
    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.mse_box = nn.MSELoss()

    def forward(self, pred, targets):
        """
        pred: [Batch, (4*reg_max + num_classes), Total_Anchors]
        targets: Liste de tenseurs [N_obj, 5] (class, x, y, w, h)
        """
        # Cette implémentation est illustrative.
        # En production, YOLO utilise un "Assigner" (ex: TAL) pour faire correspondre 
        # les prédictions aux labels réels sur la grille.
        
        # Pour cet exemple, on simule une loss sur les prédictions
        batch_size = pred.shape[0]
        loss_cls = torch.tensor(0.0, device=pred.device)
        loss_box = torch.tensor(0.0, device=pred.device)
        
        # On sépare box (4*reg_max) et cls (num_classes)
        pred_box = pred[:, :4*self.reg_max, :]
        pred_cls = pred[:, 4*self.reg_max:, :]
        
        # (Logique de matching simplifiée manquante ici pour un entraînement réel efficace)
        # Ici on retourne une loss bidon si pas de matching pour éviter le crash
        return loss_cls + loss_box

# --- Boucle d'Entraînement ---
def train_model():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 320
    batch_size = 8
    epochs = 10
    num_classes = 30 # À ajuster selon votre dataset
    
    # Chemins des données
    train_img = "./data/train/images"
    train_lbl = "./data/train/labels"
    valid_img = "./data/valid/images"
    valid_lbl = "./data/valid/labels"

    # Datasets & Dataloaders
    train_dataset = YOLODataset(train_img, train_lbl, img_size=img_size)
    val_dataset = YOLODataset(valid_img, valid_lbl, img_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Modèle
    model = YOLOv11nMobileNet(num_classes=num_classes).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = SimpleYOLOLoss(num_classes=num_classes)

    print(f"Démarrage de l'entraînement sur {device}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            # targets reste une liste (car nb d'objets variable)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Loss computation
            loss = criterion(outputs, targets)
            
            # Note : Pour l'exemple, si la loss est 0 (matching non implémenté), 
            # on fait une passe factice pour valider le code
            if loss == 0:
                loss = outputs.sum() * 0 
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Sauvegarder le modèle
        torch.save(model.state_dict(), f"yolov11n_mobile_epoch_{epoch+1}.pth")
        print(f"Fin Epoch {epoch+1}, Loss moyenne: {epoch_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    # Vérifier que les dossiers existent avant de lancer
    if os.path.exists("./data/train/images"):
        train_model()
    else:
        print("Erreur : Dossier 'data/train/images' introuvable. Vérifiez votre structure de fichiers.")
