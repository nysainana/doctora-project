import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models._utils import IntermediateLayerGetter

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """Standard convolution with BN and SiLU."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions (Standard YOLOv8)."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class DFL(nn.Module):
    """Distribution Focal Loss (DFL) module."""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data.copy_(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class MobileNetV3SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        return_layers = {'3': 'P3', '8': 'P4', '12': 'P5'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.out_channels = {'P3': 24, 'P4': 48, 'P5': 576}

    def forward(self, x):
        return self.body(x)

class YOLOv11nMobileNet(nn.Module):
    def __init__(self, num_classes=30, reg_max=16):
        super().__init__()
        self.backbone = MobileNetV3SmallBackbone()
        
        # Canaux plus robustes pour le Neck (Standard Nano)
        ch = [64, 128, 256]
        
        # Projections
        self.proj_p3 = Conv(self.backbone.out_channels['P3'], ch[0], 1)
        self.proj_p4 = Conv(self.backbone.out_channels['P4'], ch[1], 1)
        self.proj_p5 = Conv(self.backbone.out_channels['P5'], ch[2], 1)
        
        # Neck (PANet simplifié et robuste avec C2f)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down
        self.c2f_p4_td = C2f(ch[2] + ch[1], ch[1], n=1)
        self.c2f_p3_td = C2f(ch[1] + ch[0], ch[0], n=1)
        
        # Bottom-up
        self.cv_p3_p4 = Conv(ch[0], ch[0], 3, 2)
        self.c2f_p4_bu = C2f(ch[0] + ch[1], ch[1], n=1)
        
        self.cv_p4_p5 = Conv(ch[1], ch[1], 3, 2)
        self.c2f_p5_bu = C2f(ch[1] + ch[2], ch[2], n=1)
        
        # Decoupled Head (128 canaux pour plus de capacité)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.head_ch = 128 
        
        self.head_p3 = self._make_head(ch[0])
        self.head_p4 = self._make_head(ch[1])
        self.head_p5 = self._make_head(ch[2])
        
        self._init_weights()

    def _make_head(self, in_ch):
        return nn.ModuleDict({
            'box': nn.Sequential(Conv(in_ch, self.head_ch, 3), Conv(self.head_ch, self.head_ch, 3), nn.Conv2d(self.head_ch, 4 * self.reg_max, 1)),
            'cls': nn.Sequential(Conv(in_ch, self.head_ch, 3), Conv(self.head_ch, self.head_ch, 3), nn.Conv2d(self.head_ch, self.num_classes, 1))
        })

    def _init_weights(self):
        import math
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Ne pas réinitialiser les couches qui ne s'entraînent pas (comme DFL)
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        
        # Initialisation du biais pour la classification (prior_prob=0.01)
        # Aide à éviter que le modèle ne prédise n'importe quoi au début à cause du déséquilibre de classes
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        for head in [self.head_p3, self.head_p4, self.head_p5]:
            nn.init.constant_(head['cls'][-1].bias, bias_init)

    def forward(self, x):
        feats = self.backbone(x)
        p3 = self.proj_p3(feats['P3'])
        p4 = self.proj_p4(feats['P4'])
        p5 = self.proj_p5(feats['P5'])
        
        # Neck
        p4_td = self.c2f_p4_td(torch.cat([self.up(p5), p4], 1))
        p3_td = self.c2f_p3_td(torch.cat([self.up(p4_td), p3], 1))
        
        p4_bu = self.c2f_p4_bu(torch.cat([self.cv_p3_p4(p3_td), p4_td], 1))
        p5_bu = self.c2f_p5_bu(torch.cat([self.cv_p4_p5(p4_bu), p5], 1))
        
        # Heads
        results = []
        for feat, head in zip([p3_td, p4_bu, p5_bu], [self.head_p3, self.head_p4, self.head_p5]):
            box = head['box'](feat).view(feat.shape[0], 4 * self.reg_max, -1)
            cls = head['cls'](feat).view(feat.shape[0], self.num_classes, -1)
            results.append(torch.cat([box, cls], 1))
        
        return torch.cat(results, 2)
