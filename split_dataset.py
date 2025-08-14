import argparse, os, random, shutil
from pathlib import Path

def copy_samples(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dest_dir, os.path.basename(f)))

def main():
    ap = argparse.ArgumentParser(description="Split a class-folder dataset into train/val/test")
    ap.add_argument('--source_dir', required=True, help='Root with class subfolders (e.g., raw/real, raw/fake)')
    ap.add_argument('--target_dir', required=True, help='Output root (data/)')
    ap.add_argument('--train', type=float, default=0.7)
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "Splits must sum to 1.0"

    random.seed(args.seed)
    classes = [d for d in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir, d))]
    for cls in classes:
        files = []
        for root, _, fnames in os.walk(os.path.join(args.source_dir, cls)):
            for n in fnames:
                if n.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    files.append(os.path.join(root, n))
        random.shuffle(files)
        n = len(files)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        copy_samples(train_files, os.path.join(args.target_dir, 'train', cls))
        copy_samples(val_files, os.path.join(args.target_dir, 'val', cls))
        copy_samples(test_files, os.path.join(args.target_dir, 'test', cls))

    print(f"Done. Splits written to: {args.target_dir}")

if __name__ == '__main__':
    main()
