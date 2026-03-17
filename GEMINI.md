# YOLOv11n-MobileNetV3 Project Context

This project implements an ultra-lightweight object detection architecture designed for real-time inference on mobile devices (Android/iOS). It combines a **MobileNetV3-Small** backbone with a **YOLOv11n (Nano)** neck and head.

## 🏗️ Architecture Overview

- **Backbone**: `MobileNet_V3_Small` (pretrained on ImageNet). Features are extracted at three scales (P3, P4, P5) using `IntermediateLayerGetter`.
- **Neck**: Inspired by YOLOv11n (Nano), utilizing:
    - `C3k2` (CSP Bottleneck with 2 convolutions).
    - `C2PSA` (Cross-Stage Partial with Spatial Attention).
    - `PANet` / `FPN` for feature fusion.
- **Head**: Decoupled head for classification and box regression (Anchor-free).
    - **Regression**: Uses Distribution Focal Loss (DFL) with `reg_max=16`.
    - **Classification**: Binary Cross Entropy with Logits.
- **Optimization**: Targeted for < 4M parameters to ensure high performance on mobile CPUs.

## 🚀 Key Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/macOS

# Install dependencies
pip install torch torchvision onnx onnxscript pillow tqdm
```

### Training
The training script expects data in `data/train` and `data/valid` following the YOLO format.
```bash
python train.py
```
- Default image size: 320x320.
- Default epochs: 10 (configurable in `train.py`).
- Saves weights as `yolov11n_mobile_epoch_N.pth`.

### Export & Validation
Validates the architecture (parameter count check) and exports to ONNX.
```bash
python model.py
```
- Generates `yolov11n_mobilenet_v3.onnx`.

## 📱 Mobile Deployment

### Android (TFLite)
```bash
pip install onnx2tf
onnx2tf -i yolov11n_mobilenet_v3.onnx -o saved_model
```

### iOS (CoreML)
```python
import coremltools as ct
model = ct.converters.onnx.convert(model='yolov11n_mobilenet_v3.onnx')
model.save('yolov11n_mobilenet_v3.mlmodel')
```

## 🛠️ Development Conventions

- **Framework**: PyTorch.
- **Input Resolution**: Optimized for 320x320 for mobile efficiency.
- **Model Design**: Modular implementation in `model.py` using standard YOLO-like blocks (`Conv`, `Bottleneck`, `C3k2`, `C2PSA`, `DFL`).
- **Data Format**: Standard YOLO format (txt files with `class x_center y_center width height` normalized).
- **Inference**: Focus on CPU performance; avoid heavy operations or large channel counts in the neck/head.
