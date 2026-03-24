import pandas as pd
import os

from agent1 import dataset_agent  # твой парсер
from data_quality_agent import DataQualityAgent
from annotation_agent import AnnotationAgent
from al_agent import ActiveLearningAgent


TEXT_COL = "review"
LABEL_COL = "sentiment"


os.makedirs("reports", exist_ok=True)


# -----------------------------------
# STEP 1: PARSER
# -----------------------------------

def collect():
    print("\n📥 STEP 1: Dataset search")

    result = dataset_agent("sentiment analysis movie reviews")

    with open("reports/datasets.txt", "w", encoding="utf-8") as f:
        f.write(result)

    # используем локальный датасет
    df = pd.read_csv("train.csv")

    return df


# -----------------------------------
# STEP 2: CLEANING
# -----------------------------------

def clean(df):
    print("\n🧹 STEP 2: Cleaning")

    agent = DataQualityAgent()

    report = agent.detect_issues(df)
    print("Report:", report)

    df_clean = agent.fix(df, strategy={
        "missing": "median",
        "duplicates": "drop",
        "outliers": "clip_iqr"
    })

    return df_clean


# -----------------------------------
# STEP 3: AUTO LABEL
# -----------------------------------

def auto_label(df):
    print("\n🏷️ STEP 3: Auto labeling")

    agent = AnnotationAgent(modality="text")

    df_labeled = agent.auto_label(df[:100], text_col=TEXT_COL)

    return df_labeled


# -----------------------------------
# STEP 4: HUMAN-IN-THE-LOOP
# -----------------------------------

def human_review(df):
    print("\n👤 STEP 4: Human review")

    low_conf = df[df["confidence"] < 0.7]

    # 👉 если нет данных для ревью — просто пропускаем шаг
    if len(low_conf) == 0:
        print("⚠️ Нет низкой уверенности — пропускаем human review")
        return df

    # сохраняем очередь
    low_conf.to_csv("review_queue.csv", index=False)
    print(f"Сохранено {len(low_conf)} записей в review_queue.csv")

    print("Открой review_queue.csv и исправь → сохрани как review_queue_corrected.csv")
    input("Нажми Enter после проверки...")

    # 👉 если пользователь НЕ сделал файл — не ломаем pipeline
    if not os.path.exists("review_queue_corrected.csv"):
        print("⚠️ review_queue_corrected.csv не найден — пропускаем изменения")
        return df

    corrected = pd.read_csv("review_queue_corrected.csv")

    # 👉 защита от пустого файла
    if len(corrected) == 0:
        print("⚠️ review_queue_corrected.csv пуст — пропускаем изменения")
        return df

    # нормальная сборка датасета
    high_conf = df[df["confidence"] >= 0.7]
    df = pd.concat([high_conf, corrected], ignore_index=True)

    print(f"✅ После ревью: {len(df)} записей")

    return df


# -----------------------------------
# STEP 5: ACTIVE LEARNING (ОБУЧЕНИЕ ВНУТРИ)
# -----------------------------------

def train_with_al(df):
    print("\n🤖 STEP 5: Active Learning")

    df = df.sample(frac=1, random_state=42)

    n = len(df)

    labeled_df = df.iloc[:int(0.1 * n)]
    pool_df    = df.iloc[int(0.1 * n):int(0.8 * n)]
    test_df    = df.iloc[int(0.8 * n):]

    agent = ActiveLearningAgent()

    history, final_model = agent.run_cycle(
        labeled_df=labeled_df,
        pool_df=pool_df,
        test_df=test_df,
        strategy="entropy",
        text_col=TEXT_COL,
        label_col=LABEL_COL
    )

    agent.report({"entropy": history})

    return final_model


# -----------------------------------
# PIPELINE
# -----------------------------------

def run_pipeline():

    df = collect()
    df_clean = clean(df)
    df_labeled = auto_label(df_clean)
    df_reviewed = human_review(df_labeled)
    model = train_with_al(df_reviewed)

    print("\n✅ PIPELINE DONE")


if __name__ == "__main__":
    run_pipeline()