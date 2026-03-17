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

# --- Helpers ---
def dist2bbox(distance, anchor_points, stride_tensor):
    lt, rb = distance.chunk(2, 1)
    x1y1 = anchor_points - lt * stride_tensor
    x2y2 = anchor_points + rb * stride_tensor
    return torch.cat((x1y1, x2y2), 1)

def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + 1e-7
    return inter / union

def make_anchors(strides=[8, 16, 32], feats_shapes=[(40,40), (20,20), (10,10)]):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        h, w = feats_shapes[i]
        sx = torch.arange(w) + 0.5
        sy = torch.arange(h) + 0.5
        grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((grid_x, grid_y), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# --- Métriques ---
@torch.no_grad()
def calculate_batch_metrics(outputs, targets):
    device = outputs.device
    pred_scores = torch.sigmoid(outputs[:, 64:, :])
    max_scores, pred_cls = torch.max(pred_scores, dim=1)
    tp, total_gt = 0, 0
    for i in range(len(targets)):
        gt = targets[i]
        if len(gt) == 0: continue
        total_gt += len(gt)
        mask = max_scores[i] > 0.1 # Seuil bas pour le monitoring
        if not mask.any(): continue
        p_cls = pred_cls[i][mask]
        for g in gt:
            if int(g[0]) in p_cls:
                tp += 1
                break
    return (tp / (total_gt + 1e-7)) * 100

# --- Dataset ---
class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=320):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
        self.label_dir = label_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Dataset : {len(self.img_paths)} images.")

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    v = [float(x) for x in line.split()]
                    if len(v) == 5: labels.append(v)
        return self.transform(image), torch.tensor(labels) if labels else torch.zeros((0, 5))

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, 0), targets

# --- Loss ---
class RobustYOLOLoss(nn.Module):
    def __init__(self, num_classes=30, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.anchors, self.strides = make_anchors()
        self.dfl_conv = DFL(reg_max)

    def forward(self, pred, targets):
        device = pred.device
        batch_size = pred.shape[0]
        anchors, strides = self.anchors.to(device), self.strides.to(device)
        pred_dist, pred_cls = pred[:, :64, :], pred[:, 64:, :]
        loss_cls, loss_box = torch.zeros(1, device=device), torch.zeros(1, device=device)
        n_targets = 0
        for i in range(batch_size):
            target = targets[i]
            if len(target) == 0:
                loss_cls += self.bce(pred_cls[i], torch.zeros_like(pred_cls[i])) * 0.05
                continue
            n_targets += len(target)
            gt_pixels = target[:, 1:].clone()
            gt_pixels[:, [0, 2]] *= 320
            gt_pixels[:, [1, 3]] *= 320
            gt_box = torch.cat([gt_pixels[:, :2] - gt_pixels[:, 2:]/2, gt_pixels[:, :2] + gt_pixels[:, 2:]/2], 1)
            target_cls = torch.zeros_like(pred_cls[i])
            fg_mask = torch.zeros(2100, dtype=torch.bool, device=device)
            matched_gt = torch.zeros((2100, 4), device=device)
            anchor_centers = anchors * strides
            for gt_idx, gt_center in enumerate(gt_pixels[:, :2]):
                dist = torch.norm(anchor_centers - gt_center, dim=1)
                _, topk = torch.topk(dist, k=10, largest=False)
                fg_mask[topk] = True
                target_cls[target[gt_idx, 0].long(), topk] = 1.0
                matched_gt[topk] = gt_box[gt_idx]
            loss_cls += self.bce(pred_cls[i], target_cls)
            if fg_mask.any():
                p_dist = self.dfl_conv(pred_dist[i:i+1, :, fg_mask]).squeeze(0)
                p_box = dist2bbox(p_dist.T, anchors[fg_mask], strides[fg_mask])
                iou = bbox_iou(p_box, matched_gt[fg_mask])
                loss_box += (1.0 - iou).sum()
        divisor = max(n_targets, 1)
        return (loss_box / divisor), (loss_cls / divisor / 10.0), n_targets

# --- Train ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv11nMobileNet(num_classes=30).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    # Scheduler: Réduit le LR par 10 si la perte ne descend plus pendant 3 époques
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    criterion = RobustYOLOLoss(num_classes=30)
    train_loader = DataLoader(YOLODataset("./data/train/images", "./data/train/labels"), batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    for epoch in range(100): # Augmenté pour laisser le temps au nouveau modèle
        model.train()
        sum_loss, sum_acc = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            l_box, l_cls, n_obj = criterion(out, targets)
            total_loss = l_box * 5.0 + l_cls
            total_loss.backward()
            optimizer.step()
            acc = calculate_batch_metrics(out, targets)
            sum_loss += total_loss.item()
            sum_acc += acc
            pbar.set_postfix({"Loss": f"{total_loss.item():.2f}", "Acc": f"{acc:.1f}%", "Obj": n_obj})

        avg_loss = sum_loss / len(train_loader)
        avg_acc = sum_acc / len(train_loader)
        print(f"\n--- RÉSUMÉ ÉPOQUE {epoch+1} ---")
        print(f"Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.1f}%")
        
        # Mise à jour du scheduler
        scheduler.step(avg_loss)
        torch.save(model.state_dict(), f"yolov11n_mobile_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_model()
