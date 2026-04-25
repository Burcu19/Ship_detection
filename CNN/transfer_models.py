import os
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50


class TransferModel:
    """Transfer learning modelleri (feature extraction).
    YOLO hariç tüm modeller TensorFlow/Keras ile kurulur. YOLO için Ultralytics (PyTorch) kullanılır,
    çünkü Keras'ta resmi YOLOv9-cls bulunmamaktadır."""

    def __init__(self, image_size=224, num_classes=2):
        self.image_size = image_size
        self.num_classes = num_classes

    def _classifier_head(self, x):
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        return out

    def alexnet(self):
        """Keras'ta resmi pretrained AlexNet yok; klasik mimari sıfırdan kurulur."""
        inp = layers.Input(shape=(self.image_size, self.image_size, 3))
        x = layers.Conv2D(96, 11, strides=4, activation="relu", padding="same")(inp)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = layers.Conv2D(256, 5, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = layers.Conv2D(384, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(384, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D(3, strides=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inp, out, name="AlexNet")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def vgg16(self):
        base = VGG16(include_top=False, weights="imagenet",
                     input_shape=(self.image_size, self.image_size, 3))
        base.trainable = False
        inp = layers.Input(shape=(self.image_size, self.image_size, 3))
        x = base(inp, training=False)
        out = self._classifier_head(x)
        model = models.Model(inp, out, name="VGG16")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def resnet50(self):
        base = ResNet50(include_top=False, weights="imagenet",
                       input_shape=(self.image_size, self.image_size, 3))
        base.trainable = False
        inp = layers.Input(shape=(self.image_size, self.image_size, 3))
        x = base(inp, training=False)
        out = self._classifier_head(x)
        model = models.Model(inp, out, name="ResNet50")
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def yolov9(self):
        """YOLO classification (Ultralytics, PyTorch tabanlı).
        YOLOv9-cls resmi olarak yayımlanmadığı için pratikte yolov8n-cls.pt kullanılır.
        Eğitim ve tahmin akışı yolov9_train_predict() üzerinden çalıştırılır."""
        from ultralytics import YOLO
        return YOLO("yolov8n-cls.pt")

    def yolov9_prepare_dataset(self, file_paths, labels, split, root_dir):
        """Ultralytics classification formatı: root/<split>/<class>/img.png"""
        from PIL import Image
        for cls in ["0", "1"]:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)
        for p, y in zip(file_paths, labels):
            img = Image.open(p).convert("RGB").resize((self.image_size, self.image_size))
            img.save(os.path.join(root_dir, split, str(int(y)), os.path.basename(p)))

    def yolov9_train_predict(self, splits_dict, epochs=3, batch=32, project_dir="results/yolo"):
        """splits_dict: {'train': (paths, labels), 'val': (...), 'test': (...)}"""
        tmp_root = tempfile.mkdtemp(prefix="yolo_ds_")
        for split, (paths, labels) in splits_dict.items():
            self.yolov9_prepare_dataset(paths, labels, split, tmp_root)
        model = self.yolov9()
        model.train(data=tmp_root, epochs=epochs, batch=batch,
                    imgsz=self.image_size, project=project_dir, name="run", verbose=False)

        test_paths, test_labels = splits_dict["test"]
        preds = []
        for p in test_paths:
            r = model.predict(p, imgsz=self.image_size, verbose=False)[0]
            top1 = int(r.probs.top1)
            preds.append(top1)
        return np.array(preds), np.array(test_labels)
