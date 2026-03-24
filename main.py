import pandas as pd
from data_quality_agent import DataQualityAgent

df = pd.read_csv("synthetic_dataset.csv")

agent = DataQualityAgent(api_key="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")

report = agent.detect_issues(df)

df_clean = agent.fix(
    df,
    strategy={
        "missing": "median",
        "duplicates": "drop",
        "outliers": "clip_iqr"
    }
)

comparison = agent.compare(df, df_clean)

explanation = agent.explain_strategy(
    strategy={
        "missing": "median",
        "duplicates": "drop",
        "outliers": "clip_iqr"
    },
    report=report
)

print(report)
print(comparison)
print(explanation)