import argparse, os, json
from pathlib import Path

def load_hist(model_dir):
    p = os.path.join(model_dir, 'history.json')
    if not os.path.exists(p):
        return {}
    with open(p, 'r') as f:
        return json.load(f)

def load_report(model_dir):
    p = os.path.join(model_dir, 'report.txt')
    if not os.path.exists(p):
        return ""
    with open(p, 'r') as f:
        return f.read()

def extract_test_accuracy(report_text):
    # crude parse: find 'accuracy' line
    for line in report_text.splitlines():
        line = line.strip()
        if line.startswith('accuracy'):
            parts = [x for x in line.split(' ') if x]
            try:
                return float(parts[-2])
            except:
                pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dirs', nargs='+', required=True)
    args = ap.parse_args()

    rows = []
    for md in args.model_dirs:
        hist = load_hist(md)
        report = load_report(md)
        test_acc = extract_test_accuracy(report)
        best_val_acc = max(hist.get('val_accuracy', [0]))
        rows.append({
            'model_dir': md,
            'best_val_acc': round(best_val_acc, 4),
            'test_acc_from_report': test_acc
        })
    out = {'comparison': rows}
    print(out)
    Path('outputs').mkdir(exist_ok=True, parents=True)
    with open('outputs/model_comparison.json', 'w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()
