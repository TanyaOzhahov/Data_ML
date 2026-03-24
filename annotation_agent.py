from xml.parsers.expat import model

import pandas as pd
import json
from collections import Counter
from openai import OpenAI
import numpy as np

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

class AnnotationAgent:
    def __init__(self, modality='text', api_key=None, model="gpt-4o-mini"):
        self.modality = modality
    
        final_key = api_key or os.getenv("OPENAI_API_KEY")
    
        if not final_key:
            self.client = None
        else:
            self.client = OpenAI(api_key=final_key)
    
        self.model = model

    

    # ---------------------------------------------------
    # AUTO LABEL (TEXT via LLM)
    # ---------------------------------------------------

    def auto_label(self, df, text_col="text", labels=None):

        if self.modality != "text":
            raise NotImplementedError("Only text modality implemented")

        if self.client is None:
            raise ValueError("OPENAI_API_KEY not found")

        if labels is None:
            labels = ["positive", "negative", "neutral"]

        predictions = []
        confidences = []

        for text in df[text_col]:

            prompt = f"""
Ты модель для классификации текста.

Классы:
{labels}

Текст:
{text}

Ответь строго в JSON:
{{"label": "...", "confidence": 0.0}}
"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты классификатор текста."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            try:
                result = json.loads(response.choices[0].message.content)
                predictions.append(result["label"])
                confidences.append(result["confidence"])
            except:
                predictions.append("error")
                confidences.append(0.0)

        df["label_auto"] = predictions
        df["confidence"] = confidences

        return df

    # ---------------------------------------------------
    # GENERATE ANNOTATION SPEC
    # ---------------------------------------------------

    def generate_spec(self, df, task="classification", text_col="text"):

        sample_texts = df[text_col].dropna().sample(min(10, len(df))).tolist()

        prompt = f"""
Создай инструкцию для разметки данных.

Задача: {task}

Примеры текстов:
{sample_texts}

Сделай Markdown документ со структурой:

1. Описание задачи
2. Классы с определениями
3. Примеры (минимум 3 на класс)
4. Граничные случаи
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты senior data annotator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        spec = response.choices[0].message.content

        with open("annotation_spec.md", "w", encoding="utf-8") as f:
            f.write(spec)

        return spec

    # ---------------------------------------------------
    # QUALITY CHECK
    # ---------------------------------------------------

    def check_quality(self, df, human_col=None):

        metrics = {}

        # Label distribution
        dist = df["label_auto"].value_counts(normalize=True).to_dict()
        metrics["label_dist"] = dist

        # Confidence
        metrics["confidence_mean"] = float(np.mean(df["confidence"]))

        # Agreement (если есть human labels)
        if human_col and human_col in df.columns:

            agreement = (df["label_auto"] == df[human_col]).mean()
            metrics["agreement"] = float(agreement)

        return metrics

    # ---------------------------------------------------
    # EXPORT TO LABEL STUDIO
    # ---------------------------------------------------

    def export_to_labelstudio(self, df, text_col="text"):

        tasks = []

        for i, row in df.iterrows():

            task = {
                "id": int(i),
                "data": {
                    "text": row[text_col]
                },
                "predictions": [
                    {
                        "model_version": "annotation_agent",
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {
                                    "choices": [row["label_auto"]]
                                }
                            }
                        ],
                        "confidence": float(row.get("confidence", 0))
                    }
                ]
            }

            tasks.append(task)

        with open("labelstudio_import.json", "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        return tasks