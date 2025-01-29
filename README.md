# 🧠 Brain Tumor Classification using Transfer Learning

## 🛠 Implementation Details

### 🧩 Step-by-Step Approach
1. **Data Preparation**  
   - **Stratified Split**: 80% train, 20% validation (preserving class ratios)  
   - **Class Balancing**:  
     - Menin: Augmentation (rotation/flips/zoom) + class weighting  
     - Glioma: Light augmentation  
   - **Normalization**: ImageNet mean/std (`[0.485, 0.456, 0.406]`/`[0.229, 0.224, 0.225]`)  

2. **Model Development**  
   - **Base Model**: ResNet50 (pretrained on ImageNet)  
   - **Custom Head**:  
     ```python
     GlobalAveragePooling2D() → Dropout(0.5) → Dense(2, softmax)
     ```  
   - **Freezing Strategy**:  
     - Phase 1: All base layers frozen  
     - Phase 2: Last 50% layers unfrozen  

3. **Training**  
   - **Phase 1**:  
     - Optimizer: Adam (lr=0.001)  
     - Metrics: Precision/Recall focus  
   - **Phase 2**:  
     - Optimizer: Adam (lr=0.00001)  
     - Early Stopping: Patience=5  

4. **Evaluation**  
   - **Test Augmentation**: Menin-only augmentation for balance  
   - **Grad-CAM**: Visualize tumor focus areas  

---

## 🚧 Challenges & Solutions

| Challenge                          | Solution                                                                 | Impact |
|------------------------------------|--------------------------------------------------------------------------|--------|
| **Class Imbalance (1:4 Ratio)**    | - Class weights (`{0:4.0, 1:1.0}`) <br> - Menin-specific augmentation   | ↑ Recall by 22% |
| **Overfitting**                    | - Dropout (0.5) <br> - Early stopping <br> - Limited fine-tuning         | ↓ Val-Train gap by 15% |
| **Hardware Limitations**           | - Mixed-precision training <br> - Batch size optimization (32 → 16)     | ↓ Memory usage by 40% |
| **Ambiguous Tumor Regions**        | - Grad-CAM visualization <br> - Sharpness augmentation                  | ↑ Model interpretability |
| **Validation Data Leakage**        | - `shuffle=False` in validation generator <br> - Stratified split       | ↑ True generalization |
| **Medical Image Artifacts**        | - CLAHE normalization <br> - Salt-and-pepper noise reduction            | ↑ Input quality |

---

## 📊 Dataset Statistics


*Original vs Balanced Class Distribution*

| Class      | Train | Validation | Test (Original) | Test (Balanced) |
|------------|-------|------------|-----------------|-----------------|
| **Menin**  | 512   | 128        | 150             | 300             |
| **Glioma** | 2048  | 512        | 300             | 300             |

---

## 🔄 Training Process

### Phase 1: Classifier Head
 
*Key Metrics: Precision/Recall Balance*

### Phase 2: Fine-Tuning
 
*Loss Reduction Patterns*

---

## 🎯 Performance Insights

### Before/After Solutions
| Metric    | Baseline | Optimized |
|-----------|----------|-----------|
| Menin F1  | 0.68     | 0.91      |
| Epoch Time| 142s     | 89s       |
| GPU Memory| 9.8GB    | 5.2GB     |

### Error Analysis
- **False Negatives**: 12% (small tumors <5mm)  
- **False Positives**: 8% (calcification artifacts)  

---

## 🧠 Model Interpretability

 
*Model Attention Heatmap (Red=High Focus)*
