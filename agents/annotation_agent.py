"""
SKILL: Annotation Agent
━━━━━━━━━━━━━━━━━━━━━━━
INPUT : DataFrame с текстами
OUTPUT: reports/annotation_spec.md    — инструкция разметчика
        data/labelstudio_import.json  — задачи для Label Studio
        data/labeled_dataset.csv      — датасет с авторазметкой

Агент:
  1. auto_label()              — размечает тексты через LLM (gpt-4o-mini)
  2. generate_spec()           — генерирует Markdown-инструкцию разметчика
  3. export_to_labelstudio()   — формирует JSON для импорта в Label Studio
  4. check_quality()           — проверяет качество разметки
  5. run()                     — запускает полный пайплайн и сохраняет все файлы
"""

import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SPEC_PATH = "reports/annotation_spec.md"
LABELSTUDIO_PATH = "data/labelstudio_import.json"
LABELED_PATH = "data/labeled_dataset.csv"


class AnnotationAgent:

    def __init__(
        self,
        modality: str = "text",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.modality = modality
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key) if key else None

    # ──────────────────────────────────────────────
    # 1. AUTO LABEL
    # ──────────────────────────────────────────────

    def auto_label(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        labels: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Размечает каждый текст через LLM.
        Добавляет колонки 'label_auto' и 'confidence'.
        """
        if self.modality != "text":
            raise NotImplementedError("Only text modality implemented")
        if self.client is None:
            raise ValueError("OPENAI_API_KEY not found")

        if labels is None:
            labels = ["positive", "negative", "neutral"]

        labels_str = ", ".join(labels)
        predictions, confidences = [], []

        print(f"  Labeling {len(df)} samples with classes: {labels_str}")

        for i, text in enumerate(df[text_col]):
            prompt = (
                f"You are a text classifier. Classify the text into one of these classes: {labels_str}.\n\n"
                f"Text:\n{str(text)[:500]}\n\n"
                'Respond ONLY with valid JSON: {"label": "...", "confidence": 0.0}\n'
                "confidence is a float between 0 and 1. No extra text."
            )

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = response.choices[0].message.content.strip()
                # strip possible ```json fences
                raw = raw.strip("`").lstrip("json").strip()
                result = json.loads(raw)
                predictions.append(result.get("label", "error"))
                confidences.append(float(result.get("confidence", 0.0)))
            except Exception as e:
                predictions.append("error")
                confidences.append(0.0)

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(df)} done...")

        df = df.copy()
        df["label_auto"] = predictions
        df["confidence"] = confidences
        return df

    # ──────────────────────────────────────────────
    # 2. GENERATE ANNOTATION SPEC
    # ──────────────────────────────────────────────

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment classification",
        text_col: str = "text",
        labels: list[str] | None = None,
    ) -> str:
        """
        Генерирует Markdown-инструкцию для разметчиков через LLM.
        Сохраняет в reports/annotation_spec.md.
        """
        if self.client is None:
            raise ValueError("OPENAI_API_KEY not found")

        if labels is None:
            labels = ["positive", "negative", "neutral"]

        sample_texts = (
            df[text_col].dropna().sample(min(10, len(df)), random_state=42).tolist()
        )

        prompt = f"""
You are a senior data annotation specialist. Create a detailed annotation guide in Russian language.

Task: {task}
Classes: {labels}

Sample texts from the dataset:
{json.dumps(sample_texts, ensure_ascii=False, indent=2)}

Write a professional Markdown annotation guide with this exact structure:

# Инструкция по разметке данных

## 1. Описание задачи
[detailed task description]

## 2. Классы и определения
[for each class: name, definition, key indicators]

## 3. Примеры разметки
[minimum 2 examples per class with explanation]

## 4. Граничные случаи и правила
[at least 5 edge cases with correct labels]

## 5. Частые ошибки
[common mistakes to avoid]

## 6. Контакты и эскалация
[who to contact for complex cases]
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior data annotation specialist."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        spec = response.choices[0].message.content

        os.makedirs(os.path.dirname(SPEC_PATH) or ".", exist_ok=True)
        with open(SPEC_PATH, "w", encoding="utf-8") as f:
            f.write(spec)

        print(f"✅ Annotation spec → {SPEC_PATH}")
        return spec

    # ──────────────────────────────────────────────
    # 3. EXPORT TO LABEL STUDIO
    # ──────────────────────────────────────────────

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        output_path: str = LABELSTUDIO_PATH,
    ) -> list[dict]:
        """
        Формирует JSON-файл для импорта в Label Studio.
        Включает авторазметку как pre-annotation (predictions).
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        tasks = []
        for i, row in df.iterrows():
            label = str(row.get("label_auto", ""))
            conf = float(row.get("confidence", 0.0))

            task = {
                "id": int(i),
                "data": {"text": str(row[text_col])},
                "predictions": [
                    {
                        "model_version": f"annotation_agent/{self.model}",
                        "score": conf,
                        "result": [
                            {
                                "from_name": "sentiment",
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [label]},
                            }
                        ],
                    }
                ] if label and label != "error" else [],
            }
            tasks.append(task)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"✅ Label Studio JSON → {output_path}  ({len(tasks)} tasks)")
        return tasks

    # ──────────────────────────────────────────────
    # 4. QUALITY CHECK
    # ──────────────────────────────────────────────

    def check_quality(
        self,
        df: pd.DataFrame,
        human_col: str | None = None,
    ) -> dict:
        """Returns quality metrics for the auto-labeled dataset."""
        metrics: dict = {}

        # Label distribution
        metrics["label_dist"] = df["label_auto"].value_counts().to_dict()

        # Confidence stats
        metrics["confidence_mean"] = float(df["confidence"].mean())
        metrics["confidence_low_pct"] = float((df["confidence"] < 0.7).mean())
        metrics["error_rate"] = float((df["label_auto"] == "error").mean())

        # Human agreement
        if human_col and human_col in df.columns:
            valid = df[df["label_auto"] != "error"]
            metrics["agreement"] = float((valid["label_auto"] == valid[human_col]).mean())

        return metrics

    # ──────────────────────────────────────────────
    # 5. MAIN SKILL: RUN FULL PIPELINE
    # ──────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str | None = None,
        labels: list[str] | None = None,
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """
        Skill entry-point.
        Runs: auto_label → generate_spec → export_to_labelstudio → check_quality.
        Returns labeled DataFrame; saves all output files.
        """
        os.makedirs("data", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        if labels is None:
            labels = ["positive", "negative", "neutral"]

        # Sample to avoid excessive API costs
        df_sample = df.head(n_samples).copy()

        # 1. Auto-label
        print(f"\n🏷️  Auto-labeling {len(df_sample)} samples...")
        df_labeled = self.auto_label(df_sample, text_col=text_col, labels=labels)

        # 2. Quality metrics
        print("\n📊 Quality metrics:")
        metrics = self.check_quality(df_labeled, human_col=label_col)
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        # 3. Generate spec
        print("\n📝 Generating annotation spec...")
        self.generate_spec(df_labeled, text_col=text_col, labels=labels)

        # 4. Export to Label Studio
        print("\n📤 Exporting to Label Studio format...")
        self.export_to_labelstudio(df_labeled, text_col=text_col)

        # 5. Save labeled CSV
        df_labeled.to_csv(LABELED_PATH, index=False, encoding="utf-8")
        print(f"✅ Labeled CSV → {LABELED_PATH}")

        return df_labeled


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = input("Path to CSV (default: data/clean_dataset.csv): ").strip() or "data/clean_dataset.csv"
    text_col = input("Text column name (default: text): ").strip() or "text"

    df = pd.read_csv(csv_path)
    agent = AnnotationAgent()
    df_labeled = agent.run(df, text_col=text_col)
    print(f"\nDone. Labeled {len(df_labeled)} samples.")
