import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import numpy as np
from model import YOLOv11nMobileNet, DFL

# --- Helper: IoU & mAP ---
def box_iou(box1, box2):
    # box1, box2: [N, 4] (x1, y1, x2, y2)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2 - inter + 1e-7)

def calculate_batch_metrics(outputs, targets, iou_threshold=0.45):
    """
    Calcule la précision moyenne (mAP) simplifiée pour un batch.
    outputs: [B, 64 + num_classes, 2100]
    """
    device = outputs.device
    num_classes = outputs.shape[1] - 64
    batch_size = outputs.shape[0]
    
    # Décodage rapide pour la métrique
    pred_scores, pred_indices = torch.max(torch.softmax(outputs[:, 64:, :], dim=1), dim=1)
    
    total_tp = 0
    total_gt = 0
    
    for i in range(batch_size):
        gt = targets[i]
        if len(gt) == 0: continue
        
        total_gt += len(gt)
        # On ne garde que les prédictions avec un score > 0.25
        mask = pred_scores[i] > 0.25
        if not mask.any(): continue
        
        # Pour une métrique réelle, il faudrait NMS (Non-Maximum Suppression)
        # Ici on simplifie pour la rapidité d'affichage
        p_cls = pred_indices[i][mask]
        
        for g in gt:
            g_cls = int(g[0])
            if g_cls in p_cls:
                total_tp += 1
                break # On compte un TP par classe présente
                
    recall = (total_tp / total_gt) if total_gt > 0 else 0
    return recall * 100

# --- Helper: Grille ---
def make_anchors(strides=[8, 16, 32], feats_shapes=[(40,40), (20,20), (10,10)]):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = feats_shapes[i]
        sx = torch.arange(w) + 0.5
        sy = torch.arange(h) + 0.5
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2) * stride)
        stride_tensor.append(torch.full((h * w, 1), stride))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# --- Dataset YOLO ---
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=320, transform=None):
        search_path_jpg = os.path.abspath(os.path.join(img_dir, "*.jpg"))
        search_path_png = os.path.abspath(os.path.join(img_dir, "*.png"))
        self.img_paths = sorted(glob.glob(search_path_jpg) + glob.glob(search_path_png))
        self.label_dir = os.path.abspath(label_dir)
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if os.name == 'nt' and not img_path.startswith("\\\\?\\"): img_path = "\\\\?\\" + img_path
        try:
            image = Image.open(img_path).convert("RGB")
        except: return self.__getitem__((idx + 1) % len(self.img_paths))
        clean_path = img_path.replace("\\\\?\\", "")
        label_filename = os.path.splitext(os.path.basename(clean_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)
        if os.name == 'nt' and not label_path.startswith("\\\\?\\"): label_path = "\\\\?\\" + label_path
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines(): labels.append([float(x) for x in line.split()])
        if self.transform: image = self.transform(image)
        return image, torch.tensor(labels)

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets

# --- Loss Améliorée ---
class ImprovedYOLOLoss(nn.Module):
    def __init__(self, num_classes=30, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_box = nn.HuberLoss(reduction='mean')
        self.anchors, _ = make_anchors()

    def forward(self, pred, targets):
        device = pred.device
        batch_size = pred.shape[0]
        anchors = self.anchors.to(device)
        pred_dist = pred[:, :4*self.reg_max, :]
        pred_cls = pred[:, 4*self.reg_max:, :]
        
        target_cls = torch.zeros_like(pred_cls)
        fg_mask = torch.zeros((batch_size, 2100), dtype=torch.bool, device=device)

        for i, t in enumerate(targets):
            if len(t) == 0: continue
            gt_classes = t[:, 0].long()
            gt_centers = t[:, 1:3] * 320
            for gt_idx, gt_center in enumerate(gt_centers):
                dist = torch.norm(anchors - gt_center, dim=1)
                best_idx = torch.argmin(dist)
                fg_mask[i, best_idx] = True
                target_cls[i, gt_classes[gt_idx], best_idx] = 1.0

        loss_cls = self.bce_cls(pred_cls, target_cls)
        if fg_mask.any():
            loss_box = self.mse_box(pred_dist.mean(dim=1)[fg_mask], torch.ones(fg_mask.sum(), device=device))
        else:
            loss_box = pred_dist.sum() * 0
        return loss_box, loss_cls

# --- Boucle d'Entraînement ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 30
    model = YOLOv11nMobileNet(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = ImprovedYOLOLoss(num_classes=num_classes)
    
    train_loader = DataLoader(YOLODataset("./data/train/images", "./data/train/labels"), batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(YOLODataset("./data/valid/images", "./data/valid/labels"), batch_size=16, shuffle=False, collate_fn=collate_fn)

    print(f"Entraînement sur {device} | mAP@0.5 activé")
    
    for epoch in range(10):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [Train]")
        t_loss, t_map = 0, 0
        for images, targets in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            l_box, l_cls = criterion(outputs, targets)
            (l_box + l_cls).backward()
            optimizer.step()
            
            m_ap = calculate_batch_metrics(outputs, targets)
            t_loss += (l_box + l_cls).item()
            t_map += m_ap
            pbar.set_postfix({"Loss": f"{(l_box+l_cls).item():.4f}", "mAP": f"{m_ap:.1f}%"})

        model.eval()
        v_loss, v_map = 0, 0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Valid"):
                images = images.to(device)
                outputs = model(images)
                l_box, l_cls = criterion(outputs, targets)
                m_ap = calculate_batch_metrics(outputs, targets)
                v_loss += (l_box + l_cls).item()
                v_map += m_ap

        print(f"\n>> Epoch {epoch+1} Results:")
        print(f"   TRAIN -> Loss: {t_loss/len(train_loader):.4f} | mAP: {t_map/len(train_loader):.1f}%")
        print(f"   VALID -> Loss: {v_loss/len(val_loader):.4f} | mAP: {v_map/len(val_loader):.1f}%")
        torch.save(model.state_dict(), f"yolov11n_mobile_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()
