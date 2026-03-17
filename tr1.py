import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob, os
from tqdm import tqdm
from model import Model

# ------------------------
# CONFIG 🔥
# ------------------------
CONFIG = {
    "num_classes": 30,
    "epochs": 50,
    "batch_size": 8,
    "img_size": 320,
    "lr": 1e-3,
    "train_img": "D:/elysa/doctora-project/data/train/images",
    "train_label": "D:/elysa/doctora-project/data/train/labels",
    "val_img": "D:/elysa/doctora-project/data/valid/images",
    "val_label": "D:/elysa/doctora-project/data/valid/labels"}

# ------------------------
# Dataset
# ------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self,img_dir,label_dir,size):
        self.imgs = glob.glob(img_dir+"/*.jpg")
        self.label_dir = label_dir

        self.tf = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self,i):
        img = Image.open(self.imgs[i]).convert("RGB")
        img = self.tf(img)

        label_path = os.path.join(self.label_dir,
            os.path.basename(self.imgs[i]).replace(".jpg",".txt"))

        labels=[]
        if os.path.exists(label_path):
            for l in open(label_path):
                labels.append([float(x) for x in l.split()])

        return img, torch.tensor(labels)

def collate_fn(batch):
    imgs,targets = zip(*batch)
    return torch.stack(imgs), targets

# ------------------------
# Loss
# ------------------------
class Loss(nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.nc = nc

    def forward(self,pred,targets):
        B,C,N = pred.shape
        pb = pred[:,:4,:]
        pc = pred[:,4:,:]

        loss=0

        for i in range(B):
            if len(targets[i])==0: continue

            t = targets[i]
            boxes = t[:,1:5]*CONFIG["img_size"]
            cls = t[:,0].long()

            idx = torch.randint(0,N,(len(t),),device=pred.device)

            loss += ((pb[i,:,idx].T - boxes)**2).mean()

            tcls = torch.zeros_like(pc[i])
            for j,c in enumerate(cls):
                tcls[c,idx[j]] = 1

            loss += self.bce(pc[i], tcls)

        return loss

# ------------------------
# TRAIN
# ------------------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model(CONFIG["num_classes"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = Loss(CONFIG["num_classes"])

    loader = DataLoader(
        Dataset(CONFIG["train_img"], CONFIG["train_label"], CONFIG["img_size"]),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    for epoch in range(CONFIG["epochs"]):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs,targets in pbar:
            imgs = imgs.to(device)

            pred = model(imgs)
            loss = loss_fn(pred,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        torch.save(model.state_dict(), "model.pth")

if __name__=="__main__":
    train()