import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob, os
from tqdm import tqdm
from mod1 import Modele  # ton modèle MobileNetV3 + YOLOv8 style

# ------------------------ CONFIG ------------------------
CONFIG = {
    "num_classes": 30,
    "epochs": 50,
    "batch_size": 8,
    "img_size": 320,
    "lr": 1e-3,
    "train_img": "D:/elysa/doctora-project/data/train/images",
    "train_label": "D:/elysa/doctora-project/data/train/labels",
    "val_img": "D:/elysa/doctora-project/data/valid/images",
    "val_label": "D:/elysa/doctora-project/data/valid/labels",
    "checkpoint_dir": "checkpoints"
}
os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# ------------------------ Dataset ------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, size):
        self.imgs = glob.glob(img_dir + "/*.jpg")
        self.label_dir = label_dir
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        img = self.tf(img)
        label_path = os.path.join(self.label_dir, os.path.basename(self.imgs[i]).replace(".jpg", ".txt"))
        labels = []
        if os.path.exists(label_path):
            for l in open(label_path):
                parts = [float(x) for x in l.split()]
                if len(parts) == 5:
                    labels.append(parts)
        return img, torch.tensor(labels)
def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets

# ------------------------ Loss ------------------------
class Loss(nn.Module):
    def __init__(self, nc, device):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.nc = nc
        self.device = device
    def forward(self, pred, targets):
        B, C, N = pred.shape
        pb = pred[:, :4, :]
        pc = pred[:, 4:, :]
        loss = 0
        for i in range(B):
            if len(targets[i]) == 0:
                continue
            t = targets[i].to(self.device)
            boxes = t[:, 1:5] * CONFIG["img_size"]
            cls = t[:, 0].long()
            idx = torch.randint(0, N, (len(t),), device=pred.device)
            loss += ((pb[i, :, idx].T - boxes) ** 2).mean()
            tcls = torch.zeros_like(pc[i])
            for j, c in enumerate(cls):
                tcls[c, idx[j]] = 1
            loss += self.bce(pc[i], tcls)
        return loss

# ------------------------ Metrics ------------------------
def accuracy(pred, targets):
    total = 0
    correct = 0
    B, C, N = pred.shape
    pb = torch.sigmoid(pred[:, 4:, :])
    for i in range(B):
        if len(targets[i]) == 0:
            continue
        t = targets[i].to(pred.device)
        cls = t[:, 0].long()
        idx = torch.randint(0, N, (len(t),), device=pred.device)
        for j, c in enumerate(cls):
            p_class = torch.argmax(pb[i, :, idx[j]])
            if p_class == c:
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