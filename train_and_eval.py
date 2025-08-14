import argparse, os, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from utils import ensure_dir, save_json, set_seed
from data import make_datasets
from models import simple_cnn, efficientnet_b0, vgg16_tl, resnet50_tl

def build_model(name, img_size, dropout, l2, freeze_backbone):
    if name == 'simple':
        return simple_cnn(input_shape=(img_size, img_size, 3), dropout=dropout, l2=l2)
    elif name == 'efficientnet':
        return efficientnet_b0(input_shape=(img_size, img_size, 3), dropout=dropout, freeze_backbone=freeze_backbone)
    elif name == 'vgg16':
        return vgg16_tl(input_shape=(img_size, img_size, 3), dropout=dropout, freeze_backbone=freeze_backbone)
    elif name == 'resnet50':
        return resnet50_tl(input_shape=(img_size, img_size, 3), dropout=dropout, freeze_backbone=freeze_backbone)
    else:
        raise ValueError("model must be one of: simple, efficientnet, vgg16, resnet50")

def get_optimizer(name, lr):
    name = name.lower()
    if name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError("optimizer must be one of: adam, sgd, rmsprop")

def plot_confusion_matrix(cm, classes, out_path):
    import numpy as np, itertools, matplotlib.pyplot as plt
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--model', choices=['simple','efficientnet','vgg16','resnet50'], default='simple')
    ap.add_argument('--optimizer', choices=['adam','sgd','rmsprop'], default='adam')
    ap.add_argument('--img_size', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--l2', type=float, default=1e-4)
    ap.add_argument('--freeze_backbone', type=str, default='true')
    ap.add_argument('--early_stop_patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fake_names', nargs='*', default=['fake','ai','generated','synthetic'])
    ap.add_argument('--real_names', nargs='*', default=['real','human','photo','photoreal'])
    ap.add_argument('--use_class_weights', type=str, default='false')

    args = ap.parse_args()
    freeze_backbone = args.freeze_backbone.lower() == 'true'
    use_cw = args.use_class_weights.lower() == 'true'

    set_seed(args.seed)
    train_ds, val_ds, test_ds = make_datasets(
        args.data_dir, args.img_size, args.batch_size, args.fake_names, args.real_names, seed=args.seed
    )

    model = build_model(args.model, args.img_size, args.dropout, args.l2, freeze_backbone)
    model_dir = os.path.join('outputs', args.model + f"_opt-{args.optimizer}_lr-{args.lr}_bs-{args.batch_size}")
    ensure_dir(model_dir)

    opt = get_optimizer(args.optimizer, args.lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=args.early_stop_patience, restore_best_weights=True, monitor='val_accuracy', mode='max'),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_accuracy', mode='max', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, monitor='val_loss')
    ]

    class_weight = None
    if use_cw:
        # Estimate class weights from one pass of train_ds
        import numpy as np
        y_list = []
        for _, y in train_ds.take(1000):
            y_list.append(y.numpy().flatten())
        if y_list:
            y_all = np.concatenate(y_list)
            pos = y_all.sum()
            neg = len(y_all) - pos
            w0 = (pos + neg) / (2.0 * neg + 1e-8)
            w1 = (pos + neg) / (2.0 * pos + 1e-8)
            class_weight = {0: w0, 1: w1}

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, class_weight=class_weight)

    # Save history
    hist = {k:[float(x) for x in v] for k,v in history.history.items()}
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(hist, f, indent=2)

    # Evaluate on test set
    y_true, y_prob = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0).flatten()
        y_prob.extend(p.tolist())
        y_true.extend(y.numpy().flatten().tolist())
    import numpy as np
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=['real','fake'], digits=4)
    with open(os.path.join(model_dir, 'report.txt'), 'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ['real','fake'], os.path.join(model_dir, 'confusion_matrix.png'))
    try:
        plot_roc(y_true, y_prob, os.path.join(model_dir, 'roc_curve.png'))
    except Exception as e:
        print("ROC failed:", e)

    test_acc = (y_pred == y_true).mean()
    print(f"Test Accuracy: {test_acc:.4f}")
    print(report)
    print(f"Saved artifacts in: {model_dir}")

if __name__ == '__main__':
    main()
