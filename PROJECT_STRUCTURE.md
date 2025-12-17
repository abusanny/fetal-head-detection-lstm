# Project Structure

This document describes the directory structure and organization of the autism-detection-deep-learning project.

## Directory Layout

```
autism-detection-deep-learning/
│
├── README.md                    # Main project documentation
├── PROJECT_STRUCTURE.md         # This file - directory organization guide
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file (Python template)
│
├── src/                         # Source code directory
│   ├── __init__.py             # Python package initialization
│   ├── model.py                # VGG19-LSTM model architecture
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── preprocessing.py        # Image augmentation and normalization
│   ├── train.py                # Training script with K-Fold validation
│   ├── evaluate.py             # Model evaluation metrics
│   ├── predict.py              # Inference and prediction script
│   └── utils.py                # Utility functions
│
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── exploratory_analysis.ipynb    # Initial data exploration
│   ├── model_training.ipynb          # Model training workflow
│   └── results_visualization.ipynb   # Performance visualization
│
├── data/                        # Data directory (not tracked in git)
│   ├── raw/                     # Original raw data
│   ├── processed/               # Preprocessed data
│   ├── train/                   # Training dataset
│   └── test/                    # Test dataset
│
├── models/                      # Trained model checkpoints
│   ├── best_model.h5            # Best performing model
│   ├── final_model.h5           # Final trained model
│   └── model_history.pkl        # Training history (optional)
│
├── results/                     # Experiment results and outputs
│   ├── metrics/                 # Performance metrics
│   │   ├── train_metrics.json   # Training metrics
│   │   ├── test_metrics.json    # Test metrics
│   │   └── fold_metrics.json    # K-Fold cross-validation results
│   ├── plots/                   # Generated visualizations
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── training_loss.png
│   │   └── accuracy_curve.png
│   └── predictions/             # Model predictions on test data
│
├── logs/                        # Training and experiment logs
│   └── training_log.txt         # Detailed training log
│
└── config/                      # Configuration files
    ├── config.yaml              # Model and training configuration
    └── hyperparameters.json     # Hyperparameter settings
```

## Key Directories

### `/src`
Contains all source code for the project:
- **model.py**: Defines the VGG19-LSTM architecture with Focal Loss
- **data_loader.py**: Handles dataset loading and K-Fold split
- **preprocessing.py**: Image normalization and data augmentation
- **train.py**: Main training script with 5-fold cross-validation
- **evaluate.py**: Model evaluation and metrics calculation
- **predict.py**: Inference on new images

### `/notebooks`
Jupyter notebooks for interactive analysis and experimentation:
- Exploratory data analysis (EDA)
- Model development and training
- Results visualization and comparison

### `/data`
Data storage (usually excluded from git via .gitignore):
- **raw/**: Original dataset
- **processed/**: Preprocessed and augmented data
- **train/**: Training data splits
- **test/**: Test data splits

### `/models`
Stores trained model weights and checkpoints:
- Best model from cross-validation
- Final trained model
- Training history and metadata

### `/results`
Experiment results and outputs:
- **metrics/**: Numerical performance metrics (JSON format)
- **plots/**: Visualizations (confusion matrix, ROC curves, etc.)
- **predictions/**: Model predictions on test set

### `/logs`
Training and experiment logs for debugging and reproducibility

### `/config`
Configuration files for reproducible experiments:
- Model hyperparameters
- Training settings
- Data preprocessing options

## File Naming Conventions

- Python files: `snake_case.py` (e.g., `data_loader.py`)
- Configuration files: `snake_case.yaml` or `.json`
- Model files: `model_name_timestamp.h5`
- Result files: Descriptive names with date (e.g., `metrics_2024-01-15.json`)

## Development Workflow

1. **Data Preparation**: Raw data → `/data/raw/` → Preprocessing → `/data/processed/`
2. **Model Development**: Edit code in `/src/` → Test in `/notebooks/`
3. **Training**: Run `/src/train.py` → Output saved to `/models/`
4. **Evaluation**: Run `/src/evaluate.py` → Results to `/results/`
5. **Inference**: Use `/src/predict.py` for predictions on new data

## Dependencies

All required packages are listed in `requirements.txt`. Install with:
```bash
pip install -r requirements.txt
```

## Notes

- The `.gitignore` file excludes large data files, model checkpoints, and temporary files
- Use virtual environments to isolate project dependencies
- Notebook outputs should be cleared before committing to keep repository size manageable
- Results and logs can be regenerated by running training scripts
