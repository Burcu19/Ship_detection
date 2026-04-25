import os
import json
import base64
import random
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


class VLM:
    """GPT-4o mini ile zero-shot gemi sınıflandırma.
    Görseller dengeli (10 gemi + 10 gemi yok) seçilir, JSON modunda {"label": 0/1} alınır."""

    def __init__(self, data_dir, output_dir="results", n_per_class=10, seed=42):
        load_dotenv()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_per_class = n_per_class
        self.seed = seed
        self.dongu_sayisi = n_per_class * 2
        self.model_name = "gpt-4o-mini"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        os.makedirs(self.output_dir, exist_ok=True)

    def get_data(self):
        all_files = sorted(Path(self.data_dir).glob("*.png"))
        ships = [str(p) for p in all_files if os.path.basename(p).startswith("1")]
        no_ships = [str(p) for p in all_files if os.path.basename(p).startswith("0")]
        rng = random.Random(self.seed)
        rng.shuffle(ships); rng.shuffle(no_ships)
        selected = [(p, 1) for p in ships[:self.n_per_class]] + \
                   [(p, 0) for p in no_ships[:self.n_per_class]]
        rng.shuffle(selected)
        return selected

    def prompt(self):
        return (
            "Sen uzman bir uydu görüntüsü analistisin. Sana San Francisco/San Pedro körfezleri "
            "üzerinden alınmış renkli bir uydu görüntüsü verilecek. Görüntüde GEMİ bulunup "
            "bulunmadığını tespit et.\n\n"
            "Kurallar:\n"
            "- Görüntüde net bir gemi (ticari, askeri, balıkçı, yat vb. herhangi bir deniz aracı) "
            "varsa label=1.\n"
            "- Görüntüde sadece su, kara, bina, bulut, dalga vb. varsa label=0.\n"
            "- Cevabını SADECE şu JSON formatında ver: {\"label\": 0} veya {\"label\": 1}\n"
            "- Açıklama ekleme."
        )

    def config_parameter(self):
        return {
            "model": self.model_name,
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 20,
        }

    def _encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call_openai(self, image_path):
        b64 = self._encode_image(image_path)
        cfg = self.config_parameter()
        resp = self.client.chat.completions.create(
            model=cfg["model"],
            response_format=cfg["response_format"],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
            messages=[
                {"role": "system", "content": self.prompt()},
                {"role": "user", "content": [
                    {"type": "text", "text": "Bu görselde gemi var mı? JSON ile cevapla."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
        )
        content = resp.choices[0].message.content
        try:
            parsed = json.loads(content)
            return int(parsed.get("label", -1))
        except Exception:
            return -1

    def get_result(self):
        samples = self.get_data()
        results = []
        for i, (path, true_label) in enumerate(samples, 1):
            pred = self.call_openai(path)
            entry = {
                "index": i,
                "file": os.path.basename(path),
                "true_label": true_label,
                "predicted_label": pred,
                "correct": pred == true_label,
            }
            results.append(entry)
            print(f"[{i}/{len(samples)}] {entry['file']} | true={true_label} pred={pred}")
            self._save_progress(results)
        return results

    def _save_progress(self, results):
        path = os.path.join(self.output_dir, "vlm_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
