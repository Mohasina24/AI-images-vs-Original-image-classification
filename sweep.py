import argparse, itertools, os, subprocess, sys, csv, json
from pathlib import Path

def run_cfg(py, data_dir, model, optimizer, lr, batch_size, epochs, img_size, freeze_backbone, out_root):
    cmd = [
        sys.executable, py,
        '--data_dir', data_dir,
        '--model', model,
        '--optimizer', optimizer,
        '--lr', str(lr),
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--img_size', str(img_size),
        '--freeze_backbone', 'true' if freeze_backbone else 'false'
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    # derive model_dir naming convention
    model_dir = os.path.join('outputs', f'{model}_opt-{optimizer}_lr-{lr}_bs-{batch_size}')
    return model_dir

def parse_test_acc(report_path):
    if not os.path.exists(report_path):
        return None
    with open(report_path, 'r') as f:
        txt = f.read()
    # find overall accuracy line
    for line in txt.splitlines():
        if line.strip().startswith('accuracy'):
            parts = [x for x in line.split(' ') if x]
            try:
                return float(parts[-2])
            except:
                return None
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--img_size', type=int, default=128)
    ap.add_argument('--models', nargs='*', default=['simple','efficientnet','vgg16','resnet50'])
    ap.add_argument('--optimizers', nargs='*', default=['adam','sgd','rmsprop'])
    ap.add_argument('--lrs', nargs='*', type=float, default=[1e-3, 5e-4, 1e-4])
    ap.add_argument('--batch_sizes', nargs='*', type=int, default=[16, 32, 64])
    ap.add_argument('--freeze_backbone', type=str, default='true')
    args = ap.parse_args()

    freeze = args.freeze_backbone.lower() == 'true'
    results = []
    for model, opt, lr, bs in itertools.product(args.models, args.optimizers, args.lrs, args.batch_sizes):
        md = run_cfg(os.path.join('src','train_and_eval.py'), args.data_dir, model, opt, lr, bs, args.epochs, args.img_size, freeze, 'outputs')
        rep = os.path.join(md, 'report.txt')
        acc = parse_test_acc(rep)
        results.append({'model': model, 'optimizer': opt, 'lr': lr, 'batch_size': bs, 'test_accuracy': acc, 'model_dir': md})

    Path('outputs').mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join('outputs', 'sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print('Sweep complete. Results saved to', csv_path)

if __name__ == '__main__':
    main()
