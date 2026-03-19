import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models._utils import IntermediateLayerGetter

def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = Conv(c1, c2)
        self.m = nn.Sequential(*[Conv(c2, c2) for _ in range(n)])
    def forward(self, x):
        return self.m(self.cv1(x))

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        # P3: layer 3 (output_stride 8), P4: layer 8 (output_stride 16), P5: layer 12 (output_stride 32)
        self.body = IntermediateLayerGetter(m, {'3': 'p3', '8': 'p4', '12': 'p5'})
    def forward(self, x):
        return self.body(x)

class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.p3_conv = Conv(24, 96)
        self.p4_conv = Conv(48, 192)
        self.p5_conv = Conv(576, 384)
        
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Fusion descendante (FPN)
        self.c2f_p4 = C2f(384 + 192, 192, 2)
        self.c2f_p3 = C2f(192 + 96, 96, 2)

    def forward(self, feats):
        p3, p4, p5 = feats['p3'], feats['p4'], feats['p5']
        
        p3 = self.p3_conv(p3)
        p4 = self.p4_conv(p4)
        p5 = self.p5_conv(p5)
        
        # Top-down
        p4_up = self.up(p5)
        p4 = self.c2f_p4(torch.cat([p4_up, p4], 1))
        
        p3_up = self.up(p4)
        p3 = self.c2f_p3(torch.cat([p3_up, p3], 1))
        
        return p3, p4, p5

class Head(nn.Module):
    def __init__(self, c, nc):
        super().__init__()
        # Branche de classification
        self.cls_conv = nn.Sequential(Conv(c, c), Conv(c, c), nn.Conv2d(c, nc, 1))
        # Branche de boîte englobante (bbox)
        self.box_conv = nn.Sequential(Conv(c, c), Conv(c, c), nn.Conv2d(c, 4, 1))
        # Branche d'objectness (confiance)
        self.obj_conv = nn.Sequential(Conv(c, c), nn.Conv2d(c, 1, 1))

    def forward(self, x):
        return self.box_conv(x), self.obj_conv(x), self.cls_conv(x)

class Modele(nn.Module):
    def __init__(self, nc=30):
        super().__init__()
        self.nc = nc
        self.backbone = Backbone()
        self.neck = Neck()
        self.h3 = Head(96, nc)
        self.h4 = Head(192, nc)
        self.h5 = Head(384, nc)

    def forward(self, x):
        feats = self.backbone(x)

        p3, p4, p5 = self.neck(feats)
        
        outputs = []
        for p, h in zip([p3, p4, p5], [self.h3, self.h4, self.h5]):
            box, obj, cls = h(p)
            b, _, h_feat, w_feat = box.shape
            
            # On reformate pour avoir [Batch, Canaux, 2100]
            box = torch.sigmoid(box).view(b, 4, -1)
            obj = obj.view(b, 1, -1) 
            cls = cls.view(b, self.nc, -1)
            
            outputs.append(torch.cat([box, obj, cls], 1))
            
        return torch.cat(outputs, 2)
