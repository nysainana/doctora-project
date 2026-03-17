import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models._utils import IntermediateLayerGetter

# ------------------------
# Basic
# ------------------------
def autopad(k): return k // 2

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k),bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

# ------------------------
# C2f (YOLOv8 style)
# ------------------------
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2)
        self.m = nn.Sequential(*[Conv(c2,c2) for _ in range(n)])
    def forward(self,x):
        return self.m(self.cv1(x))

# ------------------------
# Backbone (MobileNet)
# ------------------------
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.body = IntermediateLayerGetter(m, {'3':'p3','8':'p4','12':'p5'})
    def forward(self,x):
        return self.body(x)

# ------------------------
# Neck (YOLOv8 style)
# ------------------------
class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.p3 = Conv(24,96)
        self.p4 = Conv(48,192)
        self.p5 = Conv(576,384)

        self.up = nn.Upsample(scale_factor=2)

        self.c2f4 = C2f(384+192,192,2)
        self.c2f3 = C2f(192+96,96,2)

    def forward(self, feats):
        p3,p4,p5 = feats['p3'],feats['p4'],feats['p5']
        p3,p4,p5 = self.p3(p3), self.p4(p4), self.p5(p5)

        p4 = self.c2f4(torch.cat([self.up(p5),p4],1))
        p3 = self.c2f3(torch.cat([self.up(p4),p3],1))

        return p3,p4,p5

# ------------------------
# Head (decoupled)
# ------------------------
class Head(nn.Module):
    def __init__(self,c,nc):
        super().__init__()
        self.cls = nn.Sequential(Conv(c,c), nn.Conv2d(c,nc,1))
        self.box = nn.Sequential(Conv(c,c), nn.Conv2d(c,4,1))
    def forward(self,x):
        return self.box(x), self.cls(x)

# ------------------------
# Model
# ------------------------
class Model(nn.Module):
    def __init__(self,nc=30):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()

        self.h3 = Head(96,nc)
        self.h4 = Head(192,nc)
        self.h5 = Head(384,nc)

    def forward(self,x):
        feats = self.backbone(x)
        p3,p4,p5 = self.neck(feats)

        outputs=[]
        for p,h in zip([p3,p4,p5],[self.h3,self.h4,self.h5]):
            box,cls = h(p)
            b,_,h,w = box.shape

            box = box.view(b,4,-1)
            cls = cls.view(b,-1,h*w)

            outputs.append(torch.cat([box,cls],1))

        return torch.cat(outputs,2)