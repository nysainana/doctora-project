import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os, sys
from pathlib import Path
from tqdm import tqdm
from mod1 import Modele  # ton modèle MobileNetV3 + YOLOv8 style

# ------------------------ CONFIG ------------------------
CONFIG = {
    "num_classes": 30,
    "epochs": 50,
    "batch_size": 8,
    "img_size": 320,
    "lr": 1e-3,
    "train_img": Path("data/train/images"),
    "train_label": Path("data/train/labels"),
    "val_img": Path("data/valid/images"),
    "val_label": Path("data/valid/labels"),
    "checkpoint_dir": Path("checkpoints")
}
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# ------------------------ Helpers ------------------------
def get_long_path(path):
    """Prépare le chemin pour Windows pour supporter les noms longs (> 260 caractères)."""
    abs_path = str(Path(path).absolute())
    if sys.platform == "win32" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path

def make_anchors(img_size=320, strides=[8, 16, 32]):
    """Génère les coordonnées (x, y) pour chaque point de la grille."""
    anchor_points = []
    for s in strides:
        h, w = img_size // s, img_size // s
        sx = torch.arange(w) + 0.5
        sy = torch.arange(h) + 0.5
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2) * s)
    return torch.cat(anchor_points)

# ------------------------ Dataset ------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, size):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.imgs = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.jpeg")) + list(self.img_dir.glob("*.png"))
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        img_path = self.imgs[i]
        long_img_path = get_long_path(img_path)
        img = Image.open(long_img_path).convert("RGB")
        img = self.tf(img)
        
        label_path = self.label_dir / (img_path.stem + ".txt")
        long_label_path = get_long_path(label_path)
        
        labels = []
        if os.path.exists(long_label_path):
            try:
                with open(long_label_path, "r") as f:
                    for l in f:
                        parts = [float(x) for x in l.split()]
                        if len(parts) == 5:
                            labels.append(parts)
            except Exception as e:
                print(f"Erreur lecture label {label_path}: {e}")
                
        return img, torch.tensor(labels)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets

# ------------------------ Loss ------------------------
class Loss(nn.Module):
    def __init__(self, nc, device):
        super().__init__()
        # On pondère énormément les exemples positifs pour compenser le déséquilibre (1 vs 2100)
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.nc = nc
        self.device = device
        self.anchors = make_anchors().to(device)

    def forward(self, pred, targets):
        B, C, N = pred.shape
        pb = pred[:, :4, :]  # box offsets
        po = pred[:, 4, :]   # objectness logits
        pc = pred[:, 5:, :]  # class logits
        
        loss_box = 0
        loss_obj = 0
        loss_cls = 0
        
        for i in range(B):
            t = targets[i].to(self.device)
            tobj = torch.zeros(N, device=self.device)
            
            if len(t) > 0:
                t_centers = t[:, 1:3] * CONFIG["img_size"]
                dists = torch.cdist(t_centers, self.anchors)
                idx = dists.argmin(dim=1) # [num_t]
                
                # Cibles pour les boîtes : le modèle doit prédire la boîte normalisée
                # On utilise sigmoid sur la prédiction pour rester entre 0 et 1
                pred_boxes = torch.sigmoid(pb[i, :, idx].T)
                loss_box += self.mse(pred_boxes, t[:, 1:5])
                
                # Cibles pour les classes
                t_cls = torch.zeros((len(t), self.nc), device=self.device)
                for j, c in enumerate(t[:, 0].long()):
                    if c < self.nc: t_cls[j, c] = 1
                loss_cls += self.bce_cls(pc[i, :, idx].T, t_cls)
                
                # Objectness : marquer les ancres matchées
                tobj[idx] = 1.0
            
            loss_obj += self.bce_obj(po[i], tobj)
            
        return (loss_box * 5.0) + (loss_obj * 1.0) + (loss_cls * 1.0)

# ------------------------ Metrics ------------------------
def accuracy(pred, targets):
    total = 0
    correct = 0
    B, C, N = pred.shape
    # Pour l'accuracy, on regarde les classes uniquement là où il devrait y avoir des objets
    pc = torch.sigmoid(pred[:, 5:, :])
    anchors = make_anchors().to(pred.device)
    
    for i in range(B):
        t = targets[i].to(pred.device)
        if len(t) == 0: continue
        t_centers = t[:, 1:3] * CONFIG["img_size"]
        dists = torch.cdist(t_centers, anchors)
        idx = dists.argmin(dim=1)
        
        for j, target in enumerate(t):
            c_target = int(target[0])
            # On prend la classe la plus probable à l'endroit matché
            p_cls = torch.argmax(pc[i, :, idx[j]])
            if p_cls == c_target:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

# ------------------------ Training ------------------------
def train(resume=False, resume_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    modele = Modele(CONFIG["num_classes"]).to(device)
    optimizer = optim.Adam(modele.parameters(), lr=CONFIG["lr"])
    loss_fn = Loss(CONFIG["num_classes"], device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_loader = DataLoader(
        Dataset(CONFIG["train_img"], CONFIG["train_label"], CONFIG["img_size"]),
        batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        Dataset(CONFIG["val_img"], CONFIG["val_label"], CONFIG["img_size"]),
        batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 7

    # Charger checkpoint si resume
    if resume and resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        modele.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Reprise de l'entraînement à l'epoch {start_epoch}")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        modele.train()
        train_loss = 0
        train_acc = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{CONFIG['epochs']}")
        for imgs, tg in pbar:
            imgs = imgs.to(device)
            pred = modele(imgs)
            loss = loss_fn(pred, tg)
            acc = accuracy(pred, tg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        modele.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for imgs, tg in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                pred = modele(imgs)
                loss = loss_fn(pred, tg)
                acc = accuracy(pred, tg)
                val_loss += loss.item()
                val_acc += acc

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Scheduler step
        scheduler.step(val_loss)
        print(f"Learning rate actuel: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save({
                "epoch": epoch,
                "model_state": modele.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print("Early stopping déclenché !")
                break

        # Sauvegarde après chaque epoch
        torch.save({
            "epoch": epoch,
            "model_state": modele.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, os.path.join(CONFIG["checkpoint_dir"], f"checkpoint_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    latest_ckpt = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    train(resume=True, resume_path=latest_ckpt)