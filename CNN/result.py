import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Result:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.entries = {}

    def add(self, model_name, metrics_dict):
        self.entries[model_name] = metrics_dict

    def save_json(self, filename="all_results.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
        return path

    def save_table(self, filename="comparison_table.csv"):
        rows = []
        for name, m in self.entries.items():
            rows.append({
                "Model": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
                "TN": m["confusion_matrix"][0][0],
                "FP": m["confusion_matrix"][0][1],
                "FN": m["confusion_matrix"][1][0],
                "TP": m["confusion_matrix"][1][1],
            })
        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        return df, path

    def plot_confusion_matrix(self, model_name, cm):
        cm = np.array(cm)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Ship (0)", "Ship (1)"])
        ax.set_yticklabels(["No Ship (0)", "Ship (1)"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - {model_name}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        path = os.path.join(self.output_dir, f"cm_{model_name}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def plot_all_confusion_matrices(self):
        for name, m in self.entries.items():
            self.plot_confusion_matrix(name, m["confusion_matrix"])

    def plot_metric_bars(self, filename="metrics_comparison.png"):
        names = list(self.entries.keys())
        metrics_keys = ["accuracy", "precision", "recall", "f1"]
        x = np.arange(len(names))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, k in enumerate(metrics_keys):
            vals = [self.entries[n][k] for n in names]
            ax.bar(x + i * width, vals, width, label=k.capitalize())
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Karşılaştırma")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120)
        plt.close(fig)
        return path
