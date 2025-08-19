
# Pneumonia Detection from Chest X-Rays (PyTorch)

> ⚠️ **Medical disclaimer:** This project is for research/education only and is **not** a medical device. Do not use it for clinical decisions.

A clean, reproducible pipeline to train and evaluate a pneumonia classifier on chest X-ray images using transfer learning (ResNet-18). Includes:
- Training & evaluation scripts
- Inference script for single images or folders
- Simple Streamlit demo app
- GitHub Actions CI (lint + tests)
- Clear, minimal repo structure

## 1) Dataset

This repo should contain an **ImageFolder** layout. You can use the popular *Chest X-Ray Images (Pneumonia)* dataset (Kermany et al., often available on Kaggle). 


## 2) Quickstart

```bash
# Clone your new GitHub repo and enter it, then:
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Train
```bash
python src/train.py   --data_dir data   --epochs 10   --batch_size 32   --lr 3e-4   --img_size 224   --pretrained
```
Artifacts are saved in `artifacts/`:
- `best_model.pt` – model weights
- `metrics.json` – final metrics
- `label_map.json` – class->index mapping

### Evaluate on Test Set
```bash
python src/train.py --data_dir data --evaluate_only --checkpoint artifacts/best_model.pt
```

### Inference
Run on a single image or directory:
```bash
python src/inference.py --checkpoint artifacts/best_model.pt --input_path path/to/image_or_dir
```

### Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

## 3) Repo Structure


## 4) Results & Metrics

During training we report:
- Accuracy
- Precision / Recall / F1 (macro & per-class)
- ROC-AUC (macro)
- Confusion matrix (saved as PNG in `artifacts/`)

> Class imbalance is handled via **WeightedRandomSampler** and per-class weights in the loss.

## 5) Reproducibility

- Seeds are fixed (`--seed`) for dataloaders and torch backends.
- Exact hyperparams are logged to `artifacts/run_config.json`.



- Add DATASET LICENSE or link in your repo if required.
- Keep the medical disclaimer visible in any UI/demo.

