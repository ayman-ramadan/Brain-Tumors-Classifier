# Brain-Tumors-Classifier

# 🧠 Brain Tumor Classification using Transfer Learning

A deep learning solution for classifying brain tumors from MRI images into Menin and Glioma categories using ResNet50 and TensorFlow/Keras.

## 📌 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Training Process](#-training-process)
- [Results](#-results)
- [FAQ](#-faq)
- [License](#-license)

## ✨ Features
- **Class Imbalance Handling**: Automatic class weighting
- **Smart Augmentation**: Targeted MRI transformations
- **Two-Phase Training**: 
  - Phase 1: Classifier head training
  - Phase 2: Full model fine-tuning
- **Comprehensive Evaluation**: Precision/Recall metrics + Confusion Matrix


## 📂 Dataset Preparation

### Directory Structure
```
brain_tumor_dataset/
├── train/
│   ├── brain_menin/    # MRI scans (Menin)
│   └── brain_glioma/   # MRI scans (Glioma)
└── test/
    ├── brain_menin/
    └── brain_glioma/
```

### Recommended Dataset
[Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  


## 🚀 Usage

### Local Execution
```bash
python brain_tumor_classifier.py \
  --train_dir path/to/train \
  --test_dir path/to/test \
  --epochs 30
```

### Kaggle Notebook
1. Upload dataset to Kaggle
2. Set paths in code:
```python
train_dir = '/kaggle/input/brain-tumors/train'
test_dir = '/kaggle/input/brain-tumors/test'
```
3. Enable GPU acceleration
4. Run all cells

## 🧠 Model Architecture

### ResNet50 Customization
  
*Modified ResNet50 Architecture*

```python
ResNet50 Base → GlobalAvgPool → Dropout(0.5) → Dense(2, softmax)
```

**Key Parameters**:
- Input Shape: `224x224x3`
- Optimizer: Adam (1e-3 → 1e-5)
- Loss: Categorical Crossentropy

## 🏋️ Training Process

### Phase 1: Classifier Head
- Epochs: 10
- Frozen Base Layers
- LR: 0.001

### Phase 2: Fine-Tuning
- Epochs: 20
- Unfreeze Top 50% Layers
- LR: 0.00001

 
*Accuracy/Loss Progression*

## 📊 Results

### Sample Output
```text
Test Accuracy: 92.4%
Test Precision: 91.2%
Test Recall: 93.8%

Classification Report:
              precision    recall  f1-score
   Menin        0.91       0.94      0.92
  Glioma        0.93       0.90      0.92
```

### Performance Metrics
| Metric    | Menin | Glioma |
|-----------|-------|--------|
| Precision | 0.91  | 0.93   |
| Recall    | 0.94  | 0.90   |
| F1-Score  | 0.92  | 0.92   |

## ❓ FAQ

**Q: How to handle different dataset structures?**  
A: Modify the `--train_dir` and `--test_dir` arguments

**Q: Can I use DenseNet instead of ResNet?**  
```python
# In build_model():
base_model = DenseNet121(weights='imagenet', include_top=False)
```

**Q: Why am I getting low recall for Menin?**  
- Increase class weight for Menin
- Add more aggressive augmentation


