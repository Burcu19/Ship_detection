import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    """CNN modeli için ön işleme adımları gerçekleşecek, kodları olabildiğinde kolay yaz."""

    def __init__(self, data_dir, image_size=224, sample_ratio=0.1, seed=42):
        self.data_dir = data_dir
        self.image_size = image_size
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.file_paths = []
        self.labels = []

    def get_data(self):
        all_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.png")))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(all_paths))
        keep = int(len(all_paths) * self.sample_ratio)
        selected = [all_paths[i] for i in idx[:keep]]
        self.file_paths = selected
        self.labels = [int(os.path.basename(p)[0]) for p in selected]
        return self.file_paths, self.labels

    def resize_image(self, image):
        return tf.image.resize(image, (self.image_size, self.image_size))

    def normalize_image(self, image):
        return tf.cast(image, tf.float32) / 255.0

    def augmented_image(self, image):
        return image

    def _load_one(self, path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=3)
        img = self.resize_image(img)
        img = self.normalize_image(img)
        return img, label

    def _build_dataset(self, paths, labels, batch_size, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=self.seed)
        ds = ds.map(self._load_one, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def train_test_validate_split(self, batch_size=32):
        if not self.file_paths:
            self.get_data()
        paths = np.array(self.file_paths)
        labels = np.array(self.labels)

        x_temp, x_test, y_temp, y_test = train_test_split(
            paths, labels, test_size=0.10, random_state=self.seed, stratify=labels
        )
        val_ratio = 0.10 / 0.90
        x_train, x_val, y_train, y_val = train_test_split(
            x_temp, y_temp, test_size=val_ratio, random_state=self.seed, stratify=y_temp
        )

        train_ds = self._build_dataset(x_train.tolist(), y_train.tolist(), batch_size, shuffle=True)
        val_ds = self._build_dataset(x_val.tolist(), y_val.tolist(), batch_size, shuffle=False)
        test_ds = self._build_dataset(x_test.tolist(), y_test.tolist(), batch_size, shuffle=False)

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "test_paths": x_test.tolist(),
            "test_labels": y_test.tolist(),
        }
