import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os, sys
import math
from pathlib import Path
from tqdm import tqdm
from mod1 import Modele 

# ------------------------ CONFIG ------------------------
CONFIG = {
    "num_classes": 30,
    "epochs": 100,
    "batch_size": 16, 
    "img_size": 320,
    "lr": 1e-3, 
    "train_img": Path("D:/elysa/doctora-project/data/train/images"),
    "train_label": Path("D:/elysa/doctora-project/data/train/labels"),
    "val_img": Path("D:/elysa/doctora-project/data/valid/images"),
    "val_label": Path("D:/elysa/doctora-project/data/valid/labels"),
    "checkpoint_dir": Path("checkpoints")
}

# ------------------------ Fonctions Géométriques ------------------------
def get_grids(size):
    grids = []
    strides = [8, 16, 32]
    for s in strides:
        g = size // s
        y, x = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
        grid = torch.stack([x + 0.5, y + 0.5], dim=-1).view(-1, 2) / g
        grids.append(grid)
    return torch.cat(grids, 0)

def bbox_iou(box1, box2):
    # CIoU Implementation pour une précision maximale
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    union = box1[:, 2] * box1[:, 3] + box2[:, 2] * box2[:, 3] - inter + 1e-7
    iou = inter / union

    # Distance des centres
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw**2 + ch**2 + 1e-7
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2)**2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2)**2) / 4
    
    # Aspect Ratio
    v = (4 / math.pi**2) * torch.pow(torch.atan(box1[:, 2] / (box1[:, 3] + 1e-7)) - torch.atan(box2[:, 2] / (box2[:, 3] + 1e-7)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    
    return iou - (rho2 / c2 + v * alpha)

# ------------------------ Dataset ------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, size, augment=False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.imgs = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png"))
        self.size = size
        self.augment = augment
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                self.norm
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                self.norm
            ])

    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img_path = self.imgs[i]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        label_path = self.label_dir / (img_path.stem + ".txt")
        labels = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for l in f:
                    parts = [float(x) for x in l.split()]
                    if len(parts) == 5: labels.append(parts)
        return img_tensor, torch.tensor(labels)

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets

# ------------------------ Loss Ultra-Basse ------------------------
class UltraLowLoss(nn.Module):
    def __init__(self, nc, device):
        super().__init__()
        # On réduit un peu le pos_weight pour éviter que la loss obj ne soit trop haute
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
        self.bce_cls = nn.BCEWithLogitsLoss(label_smoothing=0.1) # Aide la loss à descendre plus bas
        self.nc = nc
        self.device = device
        self.grids = get_grids(CONFIG["img_size"]).to(device)

    def forward(self, pred, targets):
        B, C, N = pred.shape
        p_box, p_obj, p_cls = pred[:, :4, :], pred[:, 4, :], pred[:, 5:, :]

        t_obj = torch.zeros((B, N), device=self.device)
        t_cls = torch.zeros((B, self.nc, N), device=self.device)
        t_box = torch.zeros((B, 4, N), device=self.device)
        
        loss_iou = torch.tensor(0.0, device=self.device)
        num_gts = 0

        for i in range(B):
            if len(targets[i]) == 0: continue
            t = targets[i].to(self.device)
            num_gts += len(t)
            
            dist = torch.cdist(t[:, 1:3], self.grids)
            _, idxs = torch.topk(dist, k=3, largest=False, dim=1) 
            
            for j in range(len(t)):
                target_cells = idxs[j]
                t_obj[i, target_cells] = 1.0
                t_cls[i, int(t[j, 0]), target_cells] = 1.0
                t_box[i, :, target_cells] = t[j, 1:5]
                
                # CIoU Loss (Scale Invariant)
                p_b = p_box[i, :, target_cells].T
                t_b = t[j, 1:5].repeat(3, 1)
                loss_iou += (1.0 - bbox_iou(p_b, t_b)).sum()

        loss_obj = self.bce_obj(p_obj, t_obj)
        mask = t_obj > 0
        loss_cls = self.bce_cls(p_cls.transpose(1, 2)[mask], t_cls.transpose(1, 2)[mask]) if mask.any() else torch.tensor(0.0, device=self.device)
        
        # Normalisation par le nombre d'objets pour une loss stable
        divisor = max(num_gts * 3, 1)
        total_loss = (10.0 * loss_iou / divisor) + (1.0 * loss_obj) + (1.0 * loss_cls)
        return total_loss

# ------------------------ Accuracy ------------------------
def calculate_accuracy(pred, targets):
    B, C, N = pred.shape
    device = pred.device
    grids = get_grids(CONFIG["img_size"]).to(device)
    correct, total = 0, 0
    p_obj, p_cls = torch.sigmoid(pred[:, 4, :]), torch.sigmoid(pred[:, 5:, :])
    for i in range(B):
        if len(targets[i]) == 0: continue
        t = targets[i].to(device)
        for obj in t:
            dist = torch.cdist(obj[1:3].unsqueeze(0), grids)
            _, near_idxs = torch.topk(dist, k=5, largest=False)
            near_idxs = near_idxs[0]
            found = False
            for idx in near_idxs:
                if p_obj[i, idx] > 0.4 and torch.argmax(p_cls[i, :, idx]) == int(obj[0]):
                    found = True; break
            if found: correct += 1
            total += 1
    return (correct / total * 100) if total > 0 else 0

# ------------------------ Train ------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    modele = Modele(CONFIG["num_classes"]).to(device)
    optimizer = optim.AdamW(modele.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    
    train_loader = DataLoader(Dataset(CONFIG["train_img"], CONFIG["train_label"], CONFIG["img_size"], True), batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Dataset(CONFIG["val_img"], CONFIG["val_label"], CONFIG["img_size"], False), batch_size=CONFIG["batch_size"], collate_fn=collate_fn)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG["lr"], steps_per_epoch=len(train_loader), epochs=CONFIG["epochs"])
    loss_fn = UltraLowLoss(CONFIG["num_classes"], device)
    
    best_acc = 0
    for epoch in range(CONFIG["epochs"]):
        modele.train()
        r_loss, r_acc = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, (imgs, tg) in enumerate(pbar):
            imgs = imgs.to(device)
            out = modele(imgs)
            loss = loss_fn(out, tg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modele.parameters(), max_norm=10.0) # Sécurité
            optimizer.step()
            scheduler.step()
            
            acc = calculate_accuracy(out, tg)
            r_loss += loss.item()
            r_acc += acc
            pbar.set_postfix({"loss": f"{r_loss/(i+1):.4f}", "acc": f"{r_acc/(i+1):.1f}%"})

        modele.eval()
        v_acc = sum(calculate_accuracy(modele(imgs.to(device)), tg) for imgs, tg in val_loader) / len(val_loader)
        print(f"Val Acc: {v_acc:.2f}%")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(modele.state_dict(), CONFIG["checkpoint_dir"] / "best.pth")
            print(f"--- Nouveau Record: {best_acc:.2f}% ---")

if __name__ == "__main__":
    train()
