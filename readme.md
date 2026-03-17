# YOLOv11n-MobileNetV3 (Ultra-Lightweight Detection for Mobile)

Ce projet implémente une architecture de détection d'objets ultra-légère combinant le **Backbone MobileNetV3-Small** et le **Neck/Head de YOLOv11n (Nano)**. Le modèle est optimisé pour une inférence rapide sur les processeurs de smartphones (Android/iOS).

## 📊 Dataset
Vous pouvez accéder au dataset nécessaire à l'entraînement ici : [PlantDec Dataset (Kaggle)](https://www.kaggle.com/datasets/andresmgs/plantdec)

## 🚀 Guide d'installation et d'utilisation (De A à Z)

### 1. Prérequis
- Python 3.8 ou plus récent.
- Accès à Internet pour télécharger les poids pré-entraînés du backbone.

### 2. Configuration de l'environnement
Ouvrez votre terminal dans le dossier racine du projet et suivez ces étapes :

```bash
# 1. Créer l'environnement virtuel (venv)
python -m venv venv

# 2. Activer l'environnement virtuel
# Sur Windows :
.\venv\Scripts\activate
# Sur Linux/macOS :
source venv/bin/activate

# 3. Installer les dépendances
pip install torch torchvision onnx onnxscript pillow tqdm
```

### 3. Entraînement du modèle
Le script `train.py` permet d'entraîner le modèle sur vos données situées dans le dossier `data/`.

**Structure attendue des données :**
```
data/
├── train/
│   ├── images/  # Fichiers .jpg ou .png
│   └── labels/  # Fichiers .txt (format YOLO: class x_c y_c w h)
├── valid/
│   ├── images/
│   └── labels/
```

**Lancer l'entraînement :**
```bash
python train.py
```
*Le script sauvegardera un fichier `.pth` à chaque époque (ex: `yolov11n_mobile_epoch_1.pth`).*

### 4. Test et Exportation ONNX
Une fois entraîné (ou pour tester l'architecture vide), lancez le script principal :

```bash
python model.py
```
- **Vérifie** le nombre de paramètres (doit être < 4M).
- **Exporte** le modèle au format `yolov11n_mobilenet_v3.onnx`.

## 📱 Déploiement Mobile

Une fois le fichier `.onnx` généré, vous pouvez le convertir pour votre plateforme cible :

### Pour Android (TFLite)
Utilisez `onnx2tf` pour une conversion optimisée :
```bash
pip install onnx2tf
onnx2tf -i yolov11n_mobilenet_v3.onnx -o saved_model
```

### Pour iOS (CoreML)
Utilisez `coremltools` :
```python
import coremltools as ct
model = ct.converters.onnx.convert(model='yolov11n_mobilenet_v3.onnx')
model.save('yolov11n_mobilenet_v3.mlmodel')
```

---
*Note : Pour un entraînement de production avec Target Assignment (TAL) complet, il est recommandé d'intégrer ce modèle dans la suite Ultralytics.*
