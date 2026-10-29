# Week 5: Deep Learning Application in Data Science

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify handwritten digits 0–9 using the public Digits dataset bundled with scikit-learn. The dataset contains 1,797 small 8×8 grayscale images.

## Project contents
- `Week5_Deep_Learning_Report.docx` — complete report with methodology, architecture, code, results, figures, limitations, and references
- `deep_learning.py` — reproducible training and evaluation script
- `digits_dataset.csv` — flattened image pixels and digit labels
- `evaluation_metrics.csv` — test metrics
- `digit_cnn_model.pth` — trained model weights from the included experiment
- `visualizations/` — architecture, training curves, confusion matrix, and sample predictions
- `requirements.txt` — dependencies

## Run on Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python deep_learning.py
```

## Experiment summary
- Split: stratified 70% train, 15% validation, 15% test
- Architecture: two convolutional blocks followed by a dense classifier
- Optimizer: Adam; learning rate 0.001; weight decay 0.0001
- Loss: cross-entropy
- Regularization: dropout 0.25 and validation-based best checkpoint selection
- Test accuracy in the documented run: 0.9630
- Macro F1-score: 0.9624

The dataset is suitable for a compact CPU demonstration. Results on this small 8×8 dataset should not be assumed to generalize to real-world handwriting systems.
