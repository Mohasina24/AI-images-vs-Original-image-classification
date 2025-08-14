import argparse, os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils import ensure_dir, set_seed
from data import make_datasets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model_dir', required=True, help='Path like outputs/simple or outputs/efficientnet')
    ap.add_argument('--img_size', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--fake_names', nargs='*', default=['fake','ai','generated','synthetic'])
    ap.add_argument('--real_names', nargs='*', default=['real','human','photo','photoreal'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    _, _, test_ds = make_datasets(args.data_dir, args.img_size, args.batch_size, args.fake_names, args.real_names, seed=args.seed)

    model_path = os.path.join(args.model_dir, 'best_model.keras')
    model = tf.keras.models.load_model(model_path)

    y_true, y_prob = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0).flatten()
        y_prob.extend(p.tolist())
        y_true.extend(y.numpy().flatten().tolist())
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=['real','fake'], digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', cm)

if __name__ == '__main__':
    main()
