# AI Images vs Real Image Classification (Keras/TensorFlow)

This project trains **two image classifiers** to detect whether an image is **AI-generated (fake)** or **real** and compares their performance.

- **Model 1 (from scratch):** `SimpleCNN`
- **Model 2 (transfer learning):** `EfficientNetB0`
- **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **Training safety:** early stopping + model checkpoint + learning-rate scheduling
- **Image size:** 128×128 by default (configurable)
- **Augmentation:** flips/rotations/zoom/contrast/etc.
- **Reproducible splits:** 70/15/15 (train/val/test)

---

## 1) Dataset

Use the Kaggle dataset **"AI Generated Images vs Real Images"** (or any similar dataset organized by class folders).  
Examples on Kaggle:
- *AI Generated Images vs Real Images* by Cash Bowman  
- *CIFAKE: Real and AI-Generated Synthetic Images*  
- *AI vs. Human-Generated Images* (Women in AI 2025 challenge dataset)

> Download via Kaggle website or Kaggle API.

### Expected folder structure (after splitting)

```
data/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

You can change the **class folder names** (e.g., `ai`, `generated` instead of `fake`) using CLI flags.

> If your dataset currently has a single folder per class (e.g., `raw/real`, `raw/fake`), use `split_dataset.py` below to create the 70/15/15 split automatically.

---

## 2) Quick Start (Local: VS Code / Terminal)

### A. Create environment & install deps
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### B. Put/prepare data
Place your images into `data/` or run the split helper:

```bash
# Example: starting from raw folder structure
# raw/
#   real/
#   fake/
python src/split_dataset.py --source_dir raw --target_dir data --train 0.7 --val 0.15 --test 0.15 --seed 42
```

### C. Train (choose model: `simple` or `efficientnet`)
```bash
# Simple CNN from scratch
python src/train_and_eval.py --data_dir data --model simple --epochs 25

# EfficientNetB0 transfer learning
python src/train_and_eval.py --data_dir data --model efficientnet --epochs 15
```

Key optional flags:
```
--img_size 128  --batch_size 32
--fake_names fake ai generated # recognized folder names for "fake" class
--real_names real human photoreal photo
--lr 0.001  --freeze_backbone true/false  --early_stop_patience 5
```

### D. Evaluate on the **test** split
```bash
# Evaluate the best checkpoint saved during training
python src/evaluate.py --data_dir data --model_dir outputs/simple  --fake_names fake ai generated --real_names real human
python src/evaluate.py --data_dir data --model_dir outputs/efficientnet --fake_names fake ai generated --real_names real human
```

Outputs:
- `outputs/<model_name>/best_model.keras` – best weights
- `outputs/<model_name>/history.json` – learning curves
- `outputs/<model_name>/report.txt` – precision/recall/F1 per class & overall
- `outputs/<model_name>/confusion_matrix.png`
- `outputs/<model_name>/roc_curve.png`

### E. Compare models
```bash
python src/compare_models.py --model_dirs outputs/simple outputs/efficientnet
```
This prints a concise table and writes `outputs/model_comparison.json`.

---

## 3) Colab Tip (Kaggle API)

1. Create a Kaggle API token (`kaggle.json`) from your Kaggle account.
2. In Colab:
```python
from google.colab import files; files.upload()  # upload kaggle.json
!pip install -q kaggle
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p /content
!unzip -q /content/ai-generated-images-vs-real-images.zip -d /content/raw
```
3. Then run `split_dataset.py` as above.

---

## 4) Grading Criteria Mapping

- **Data Preprocessing (20%)**: `split_dataset.py`, normalization, augmentation, correct 70/15/15 splits.
- **Model Dev & Training (30%)**: Two models, tuned hyperparameters, early stopping, LR scheduler, checkpoints, class weights (optional).
- **Model Evaluation (20–30%)**: Validation curves, test accuracy, precision/recall/F1, confusion matrix, ROC-AUC.
- **Optimization & Generalization (20–30%)**: Transfer learning (EfficientNet), dropout, L2, augmentation, freeze/unfreeze strategy, LR tuning.

---

## 5) Troubleshooting

- If you see class name mismatches, pass your folder names via `--fake_names` and `--real_names`.
- For imbalanced data, add `--use_class_weights true` when training.
- GPU strongly recommended for EfficientNet. On CPU, use fewer epochs or smaller `--batch_size`.


## Optimization & Model Comparison

Run a quick sweep to test different **architectures**, **optimizers**, **learning rates**, and **batch sizes**:

```bash
python src/sweep.py --data_dir data --epochs 8 --img_size 128   --models simple efficientnet vgg16 resnet50   --optimizers adam sgd rmsprop   --lrs 0.001 0.0005 0.0001   --batch_sizes 16 32 64 --freeze_backbone true
```

Then compare best runs using the generated `outputs/sweep_results.csv` and the per-run `report.txt` files.

**Fine-tuning tip:** After training a transfer model with a frozen backbone, re-run with `--freeze_backbone false` and a smaller `--lr` (e.g., `1e-4`) for 5–10 more epochs.
