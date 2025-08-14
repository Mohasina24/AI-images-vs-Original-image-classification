import os, json, math, random
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int = 42):
    import numpy as np, random, tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def gather_model_paths(model_dir: str):
    best = os.path.join(model_dir, "best_model.keras")
    hist = os.path.join(model_dir, "history.json")
    return best, hist
