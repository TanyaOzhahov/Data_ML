import pandas as pd
from annotation_agent import AnnotationAgent

df = pd.read_csv("train.csv")
df = df.head(10)

agent = AnnotationAgent(
    modality="text",
    api_key="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"
)

# 1. Авторазметка
df_labeled = agent.auto_label(df, text_col="review")

# 2. Спецификация
spec = agent.generate_spec(df, task="sentiment_classification", text_col="review")

# 3. Метрики
metrics = agent.check_quality(df_labeled)

# 4. Экспорт в Label Studio
agent.export_to_labelstudio(df_labeled, text_col="review")

print(metrics)