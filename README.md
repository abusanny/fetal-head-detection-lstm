# Fetal Head Detection using VGG19-LSTM with K-Fold Cross-Validation

## Overview

This repository contains a deep learning model for **fetal head detection** in ultrasound image sequences using VGG19 convolutional neural network combined with LSTM (Long Short-Term Memory) layers. The model performs binary classification (Head vs. Not-Head) on sequential ultrasound frames using K-Fold cross-validation (5-fold) for robust evaluation and improved generalization.

## Project Details

### Model Architecture

**VGG19-LSTM with K-Fold Validation**
- **Feature Extraction**: VGG19 pre-trained on ImageNet (TimeDistributed wrapper for sequences)
- **Temporal Modeling**: LSTM layers (256 units → 128 units) for capturing temporal dependencies
- **Loss Function**: Focal Loss (alpha=0.79, gamma=2.0) for handling severe class imbalance
- **K-Fold Configuration**: 5-fold cross-validation for robust model evaluation
- **Optimization**: Adam optimizer (learning rate: 1e-4)
- **Regularization**: Dropout (0.5), Batch Normalization, L2 regularization

### Dataset

**Fetal Ultrasound Phantom Data**
- **Total Images**: 24,036 frames across 6 phantom scans (Scan 5S, 10S, 15S, 20S, 25S, 30S)
- **Training Set**: ~19,000 frames with 5,223 Head labels, 18,813 Not-Head labels
- **Class Distribution**: Severely imbalanced (0.22 positive ratio)
- **Image Preprocessing**:
  - Crop box: [188, 36, 960, 620] (left, top, right, bottom)
  - Resize to 224×224 pixels (VGG19 requirement)
  - Padding to square aspect ratio
  - Normalization to [0, 1] range
- **Sequence Length**: 8 consecutive frames per sequence (SEQLEN=8)
- **Sequence Stride**: 1 frame overlap (sliding window)
- **Batch Size**: 8 sequences per batch

### Model Performance

**Across All 5 Folds**
- **Validation Accuracy**: ~79-82% per fold
- **ROC-AUC**: ~0.79-0.85 across folds
- **PR-AUC**: ~0.21-0.60 (varies due to class imbalance)
- **Cross-Validation Strategy**: Stratified K-Fold ensures balanced fold distribution
- **Best Fold Performance**: ~84% validation accuracy with improved recall

### Technical Highlights

✅ **Handles Class Imbalance**: Focal Loss with α=0.79, γ=2.0
✅ **Temporal Modeling**: LSTM captures motion and sequential patterns
✅ **Robust Evaluation**: 5-Fold CV prevents data leakage
✅ **GPU Optimized**: Tested on NVIDIA GPU (RTX A2000)
✅ **Early Stopping**: Prevents overfitting with patience=20 epochs

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 12GB+ GPU memory recommended

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/abusanny/autism-detection-deep-learning.git
   cd autism-detection-deep-learning
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

See `PROJECT_STRUCTURE.md` for detailed directory organization.

```
├── src/
│   ├── model.py                # VGG19-LSTM architecture
│   ├── data_loader.py          # K-Fold data preparation
│   ├── preprocessing.py        # Image augmentation & normalization
│   ├── train.py                # Training script with 5-Fold CV
│   ├── evaluate.py             # Performance metrics
│   └── predict.py              # Inference on new frames
├── notebooks/
│   └── Head_Not_Head_VGG19_LSTM_KFold_5Fold.ipynb
├── results/                    # Metrics, plots, predictions
├── models/                     # Trained model checkpoints
└── requirements.txt
```

## Usage

### Training with 5-Fold Cross-Validation

```python
python src/train.py \
  --epochs 50 \
  --batch_size 8 \
  --folds 5 \
  --alpha 0.79 \
  --gamma 2.0 \
  --learning_rate 1e-4
```

### Making Predictions on New Ultrasound Frames

```python
python src/predict.py \
  --model models/best_model.h5 \
  --sequence_dir path/to/ultrasound/frames/ \
  --output results/predictions/
```

### Evaluating Model Performance

```python
python src/evaluate.py \
  --model models/best_model.h5 \
  --test_data results/fold_metrics.json
```

## Key Technologies

- **TensorFlow 2.10+**: Deep learning framework
- **Keras**: High-level API for model building
- **Python 3.8+**: Programming language
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: ML utilities (K-Fold, metrics)
- **OpenCV**: Image processing
- **Matplotlib/Seaborn**: Visualization

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Epochs | 50 (with early stopping) |
| Batch Size | 8 |
| Focal Loss α | 0.79 |
| Focal Loss γ | 2.0 |
| Dropout | 0.5 |
| K-Fold Splits | 5 |
| Early Stopping Patience | 20 epochs |

### Training Configuration

- **Input Shape**: (Batch, 8 frames, 224×224, 3 channels)
- **Output**: Binary classification (Head=1, Not-Head=0)
- **Validation Strategy**: 5-Fold cross-validation (80% train, 20% validation per fold)
- **GPU**: NVIDIA RTX A2000 12GB

## Results Interpretation

### Per-Fold Metrics

Each fold generates:
- `fold_metrics.json`: Accuracy, precision, recall, ROC-AUC, PR-AUC
- `confusion_matrix_foldX.png`: Visualization of true/false positives
- `classification_report_foldX.txt`: Detailed per-class metrics

### Aggregated Results

- `aggregated_metrics_summary.csv`: Mean ± Std across 5 folds
- `confusion_matrix_aggregated.png`: Sum of all fold confusion matrices
- `metrics_across_folds.png`: Bar plots comparing fold performance

## Dataset Citation

**Fetal Ultrasound Phantom Dataset**
- 6 different phantom scans (varying image quality and resolution)
- 24,036 annotated frames with ground truth labels (Head/Not-Head)
- Collected from clinical ultrasound systems
- Used for medical image analysis research

## Future Improvements

- 🔄 **Attention Mechanisms**: Add spatial/temporal attention layers
- 🤖 **Ensemble Methods**: Combine multiple architectures (EfficientNet, ResNet)
- 📊 **Uncertainty Quantification**: Bayesian approaches for confidence scores
- 🚀 **Model Deployment**: TensorFlow Lite for edge devices
- 🏥 **Clinical Validation**: Testing on real clinical ultrasound data
- 🎯 **Multi-task Learning**: Simultaneous head detection + measurement

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## License

This project is open source and available under the MIT License.

## References

- Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition" (VGG19)
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
- Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection"
- He, K., et al. (2015). "Deep Residual Learning for Image Recognition" (ResNet architecture reference)
- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization"

## Contact & Support

For questions, suggestions, or collaboration opportunities:
- **Email**: abusanny@github.com
- **GitHub**: https://github.com/abusanny
- **Institution**: Utsaah Lab, IIT Jodhpur

---

**Note**: This project demonstrates the application of deep learning for medical ultrasound image analysis, specifically for fetal head detection in sequential frames. The K-Fold cross-validation approach ensures robust model evaluation and generalization to unseen data.
