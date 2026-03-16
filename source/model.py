import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models._utils import IntermediateLayerGetter

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with Batch Normalization and SiLU activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(nn.Module):
    """C3k is a CSP bottleneck with 3 convolutions and k-size kernels."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2(nn.Module):
    """Faster version of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1, 1)
        self.m = nn.ModuleList(C3k(c_, c_, 2, shortcut, g, e=1.0, k=3) if c3k else Bottleneck(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class PSALayer(nn.Module):
    """Polarized Self-Attention Layer."""
    def __init__(self, c, n=4):
        super().__init__()
        self.c = c
        self.cv1 = Conv(c, c, 1, 1)
        self.cv2 = Conv(c, c, 1, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // n, 1, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(c // n, c, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attn(self.cv1(x)) + self.cv2(x)

class C2PSA(nn.Module):
    """CSP with PSA (Cross-Stage Partial with Spatial Attention)."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(PSALayer(c_) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))

class DFL(nn.Module):
    """Distribution Focal Loss (DFL) module."""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data.copy_(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class MobileNetV3SmallBackbone(nn.Module):
    """MobileNetV3-Small backbone with feature extraction at P3, P4, P5."""
    def __init__(self):
        super().__init__()
        # Load pre-trained MobileNetV3-Small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        
        # Identify layers for P3 (stride 8), P4 (stride 16), P5 (stride 32)
        # MobileNetV3-Small features structure:
        # 0: ConvBNActivation (s2) - 160x160
        # 1: InvertedResidual (s2) - 80x80 (P2)
        # 2: InvertedResidual (s1)
        # 3: InvertedResidual (s2) - 40x40 (P3) -> layer '3'
        # 4-8: InvertedResiduals (s2 at layer 8) - 20x20 (P4) -> layer '8'
        # 9-12: InvertedResiduals (s2 at layer 12) - 10x10 (P5) -> layer '12'
        
        return_layers = {
            '3': 'P3',
            '8': 'P4',
            '12': 'P5'
        }
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        # Output channels for MobilenetV3-Small at these layers:
        # P3: 24 channels
        # P4: 48 channels
        # P5: 576 channels (last block of features)
        self.out_channels = {'P3': 24, 'P4': 48, 'P5': 576}

    def forward(self, x):
        return self.body(x)

class YOLOv11nMobileNet(nn.Module):
    """YOLOv11n with MobileNetV3-Small Backbone for Mobile Devices."""
    def __init__(self, num_classes=80, reg_max=16):
        super().__init__()
        self.backbone = MobileNetV3SmallBackbone()
        
        # Width/Depth scaling for YOLOv11n (Nano)
        # We project backbone features to Nano-sized channels
        ch = [64, 128, 256] # Nano-ish channels for P3, P4, P5
        
        # 1x1 Convolutions to match Neck dimensions
        self.proj_p3 = Conv(self.backbone.out_channels['P3'], ch[0], 1)
        self.proj_p4 = Conv(self.backbone.out_channels['P4'], ch[1], 1)
        self.proj_p5 = Conv(self.backbone.out_channels['P5'], ch[2], 1)
        
        # Neck (PANet / FPN)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Top-down
        self.c3k2_p4_td = C3k2(ch[2] + ch[1], ch[1], n=1, shortcut=False)
        self.c3k2_p3_td = C3k2(ch[1] + ch[0], ch[0], n=1, shortcut=False)
        
        # Bottom-up
        self.conv_p3_p4_bu = Conv(ch[0], ch[0], 3, 2)
        self.c3k2_p4_bu = C3k2(ch[0] + ch[1], ch[1], n=1, shortcut=False)
        
        self.conv_p4_p5_bu = Conv(ch[1], ch[1], 3, 2)
        self.c3k2_p5_bu = C3k2(ch[1] + ch[2], ch[2], n=1, shortcut=False)
        
        # C2PSA (Cross-Stage Partial with Spatial Attention) - Applied at the end of Neck (P5)
        self.c2psa = C2PSA(ch[2], ch[2], n=1)
        
        # Decoupled Head
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.no = num_classes + 4 * reg_max
        
        # Separate branches for each scale
        self.head_p3 = self._make_head(ch[0])
        self.head_p4 = self._make_head(ch[1])
        self.head_p5 = self._make_head(ch[2])
        
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def _make_head(self, in_ch):
        # Decoupled head: small internal channels (64) for mobile efficiency
        internal_ch = 64
        return nn.ModuleDict({
            'box': nn.Sequential(Conv(in_ch, internal_ch, 3), Conv(internal_ch, internal_ch, 3), nn.Conv2d(internal_ch, 4 * self.reg_max, 1)),
            'cls': nn.Sequential(Conv(in_ch, internal_ch, 3), Conv(internal_ch, internal_ch, 3), nn.Conv2d(internal_ch, self.num_classes, 1))
        })

    def forward(self, x):
        # Backbone
        feats = self.backbone(x)
        p3 = self.proj_p3(feats['P3'])
        p4 = self.proj_p4(feats['P4'])
        p5 = self.proj_p5(feats['P5'])
        
        # Neck: Top-down
        p5_up = self.up(p5)
        p4_td = self.c3k2_p4_td(torch.cat([p5_up, p4], 1))
        
        p4_td_up = self.up(p4_td)
        p3_td = self.c3k2_p3_td(torch.cat([p4_td_up, p3], 1))
        
        # Neck: Bottom-up
        p3_bu = self.conv_p3_p4_bu(p3_td)
        p4_bu = self.c3k2_p4_bu(torch.cat([p3_bu, p4_td], 1))
        
        p4_bu_bu = self.conv_p4_p5_bu(p4_bu)
        p5_bu = self.c3k2_p5_bu(torch.cat([p4_bu_bu, p5], 1))
        
        p5_final = self.c2psa(p5_bu)
        
        # Heads
        results = []
        for feat, head in zip([p3_td, p4_bu, p5_final], [self.head_p3, self.head_p4, self.head_p5]):
            box = head['box'](feat)
            cls = head['cls'](feat)
            # Reshape for DFL and output format
            b, _, h, w = box.shape
            box = box.view(b, 4 * self.reg_max, -1)
            cls = cls.view(b, self.num_classes, -1)
            results.append(torch.cat([box, cls], 1))
        
        # Concatenate all scales: [Batch, (4*reg_max + num_classes), Total_Anchors]
        return torch.cat(results, 2)

# --- Test Script ---
if __name__ == "__main__":
    # Parameters
    num_classes = 80
    model = YOLOv11nMobileNet(num_classes=num_classes)
    
    # Dummy input (mobile-friendly resolution)
    dummy_input = torch.randn(1, 3, 320, 320)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {total_params:.2f} M")
    
    if total_params < 4.0:
        print("Success: Model is ultra-lightweight (< 4M parameters).")
    else:
        print("Warning: Model exceeds 4M parameters.")

    # --- Export to ONNX ---
    print("\nExporting to ONNX...")
    onnx_file = "yolov11n_mobilenet_v3.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_file, 
        opset_version=18, 
        input_names=['images'], 
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch'}, 'output': {0: 'batch'}}
    )
    print(f"Model exported to {onnx_file}")
    
    # Brief guide for mobile conversion:
    # 1. TensorFlow Lite (Android):
    #    Use 'onnx2tf' or 'onnx-tensorflow' to convert ONNX -> SavedModel -> TFLite.
    #    Example: onnx2tf -i yolov11n_mobilenet_v3.onnx -o saved_model
    # 2. CoreML (iOS):
    #    Use 'coremltools' in Python.
    #    Example: 
    #    import coremltools as ct
    #    mlmodel = ct.converters.onnx.convert(model='yolov11n_mobilenet_v3.onnx')
    #    mlmodel.save('yolov11n_mobilenet_v3.mlmodel')
