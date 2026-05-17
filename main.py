import os
import numpy as np
import tensorflow as tf

from CNN.data_preprocessing import DataPreprocessing
from CNN.transfer_models import TransferModel
from CNN.metrics import Metrics as CNNMetrics
from CNN.result import Result
from vlm.vlm import VLM
from vlm.metrics_vlm import Metrics as VLMMetrics


def _resolve_data_dir():
    candidates = [
        "data/shipsnet/shipsnet",
        "../../../data/shipsnet/shipsnet",
        os.path.join(os.path.dirname(__file__), "data/shipsnet/shipsnet"),
        os.path.join(os.path.dirname(__file__), "../../../data/shipsnet/shipsnet"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError("ShipsNet data klasoru bulunamadi.")


DATA_DIR = _resolve_data_dir()
IMAGE_SIZE = 224
SAMPLE_RATIO = 0.10
BATCH_SIZE = 32
EPOCHS = 3
RESULTS_DIR = "results"


def predict_keras(model, test_ds):
    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0).flatten()
        y_pred.extend((p >= 0.5).astype(int).tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def run_cnn(result_collector):
    print("\n=== CNN Pipeline ===")
    dp = DataPreprocessing(DATA_DIR, image_size=IMAGE_SIZE, sample_ratio=SAMPLE_RATIO)
    splits = dp.train_test_validate_split(batch_size=BATCH_SIZE)

    tm = TransferModel(image_size=IMAGE_SIZE, num_classes=2)

    keras_models = {
        "AlexNet": tm.alexnet(),
        "VGG16": tm.vgg16(),
        "ResNet50": tm.resnet50(),
    }
    for name, model in keras_models.items():
        print(f"\n--- {name} eğitimi ---")
        model.fit(splits["train"], validation_data=splits["val"],
                  epochs=EPOCHS, verbose=2)
        y_true, y_pred = predict_keras(model, splits["test"])
        m = CNNMetrics(y_true, y_pred).all()
        result_collector.add(name, m)
        print(f"{name}: acc={m['accuracy']:.3f} f1={m['f1']:.3f}")

    print("\n--- YOLOv9 (Ultralytics) eğitimi ---")
    paths = np.array(dp.file_paths); labels = np.array(dp.labels)
    from sklearn.model_selection import train_test_split
    x_temp, x_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=0.10, random_state=dp.seed, stratify=labels)
    val_ratio = 0.10 / 0.90
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_ratio, random_state=dp.seed, stratify=y_temp)

    splits_dict = {
        "train": (x_train.tolist(), y_train.tolist()),
        "val": (x_val.tolist(), y_val.tolist()),
        "test": (x_test.tolist(), y_test.tolist()),
    }
    yolo_pred, yolo_true = tm.yolov9_train_predict(
        splits_dict, epochs=EPOCHS, batch=BATCH_SIZE,
        project_dir=os.path.join(RESULTS_DIR, "yolo"))
    m = CNNMetrics(yolo_true, yolo_pred).all()
    result_collector.add("YOLOv9", m)
    print(f"YOLOv9: acc={m['accuracy']:.3f} f1={m['f1']:.3f}")


def run_vlm(result_collector):
    print("\n=== VLM Pipeline (GPT-4o mini, Zero-Shot) ===")
    vlm = VLM(DATA_DIR, output_dir=RESULTS_DIR, n_per_class=10)
    results = vlm.get_result()
    valid = [r for r in results if r["predicted_label"] in (0, 1)]
    y_true = [r["true_label"] for r in valid]
    y_pred = [r["predicted_label"] for r in valid]
    m = VLMMetrics(y_true, y_pred).all()
    result_collector.add("VLM_GPT4o-mini", m)
    print(f"VLM: acc={m['accuracy']:.3f} f1={m['f1']:.3f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rc = Result(output_dir=RESULTS_DIR)

    run_cnn(rc)
    run_vlm(rc)

    rc.save_json()
    df, csv_path = rc.save_table()
    rc.plot_all_confusion_matrices()
    rc.plot_metric_bars()

    print("\n=== KARŞILAŞTIRMA TABLOSU ===")
    print(df.to_string(index=False))
    print(f"\nSonuçlar: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
