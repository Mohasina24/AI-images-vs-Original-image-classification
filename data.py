import os
from typing import List, Tuple
import tensorflow as tf

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

def _find_class_dirs(root: str):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def map_class_names(root: str, fake_names: List[str], real_names: List[str]) -> Tuple[dict, list]:
    fake_alias = set([n.lower() for n in fake_names])
    real_alias = set([n.lower() for n in real_names])

    class_dirs = _find_class_dirs(root)
    class_map = {}
    for c in class_dirs:
        cl = c.lower()
        if cl in fake_alias:
            class_map[c] = 'fake'
        elif cl in real_alias:
            class_map[c] = 'real'
        else:
            # Heuristic: if 'fake'/'ai' in name -> fake, else real
            class_map[c] = 'fake' if any(k in cl for k in ['fake','ai','gen','synthetic']) else 'real'
    classes = sorted(list(set(class_map.values())))
    assert set(classes) == {'fake','real'}, f"Could not resolve classes into fake/real. Found: {classes}"
    return class_map, classes

def make_datasets(data_dir: str, img_size: int, batch_size: int,
                  fake_names: List[str], real_names: List[str], seed: int = 42):

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    class_map, _ = map_class_names(train_dir, fake_names, real_names)

    def dataset_from_directory(dir_path):
        return tf.keras.preprocessing.image_dataset_from_directory(
            dir_path,
            labels='inferred',
            label_mode='binary',  # 0/1
            class_names=sorted(list(class_map.keys())),
            image_size=(img_size, img_size),
            batch_size=batch_size,
            shuffle=True,
            seed=seed
        )

    train_ds = dataset_from_directory(train_dir)
    val_ds = dataset_from_directory(val_dir)
    test_ds = dataset_from_directory(test_dir)

    # Map original class order to binary 0/1 with 1 = fake, 0 = real
    # We need to convert labels if directory classes map differently.
    inv = {k:v for k,v in class_map.items()}
    # directory provides labels by index of class_names; convert: fake->1, real->0
    fake_indices = [i for i,c in enumerate(sorted(list(class_map.keys()))) if class_map[c]=='fake']
    real_indices = [i for i,c in enumerate(sorted(list(class_map.keys()))) if class_map[c]=='real']

    def to_binary(x, y):
        # y is shape (batch,1) with class indices; turn into 1 if in fake_indices else 0
        y = tf.cast(y, tf.int32)
        y = tf.squeeze(y, axis=-1)
        y_bin = tf.where(tf.reduce_any(tf.equal(tf.expand_dims(y,-1), tf.constant(fake_indices)), axis=-1), 1, 0)
        y_bin = tf.cast(tf.expand_dims(y_bin, -1), tf.float32)
        return x, y_bin

    train_ds = train_ds.map(to_binary)
    val_ds = val_ds.map(to_binary)
    test_ds = test_ds.map(to_binary)

    # Cache/shuffle/prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds
