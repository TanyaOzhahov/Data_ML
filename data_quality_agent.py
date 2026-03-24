import pandas as pd
import numpy as np
from openai import OpenAI


class DataQualityAgent:

    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Agent for detecting and fixing data quality issues.
        """

        self.client = None
        self.model = model

        if api_key:
            self.client = OpenAI(api_key=api_key)

    # ---------------------------------------------------
    # DETECT ISSUES
    # ---------------------------------------------------

    def detect_issues(self, df: pd.DataFrame):

        report = {}

        # Missing values
        missing = df.isnull().sum()
        report["missing"] = missing[missing > 0].to_dict()

        # Duplicates
        report["duplicates"] = int(df.duplicated().sum())

        # Outliers using IQR
        outliers = {}
        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            mask = (df[col] < lower) | (df[col] > upper)
            count = int(mask.sum())

            if count > 0:
                outliers[col] = count

        report["outliers"] = outliers

        # Class imbalance
        imbalance = {}
        cat_cols = df.select_dtypes(include="object").columns

        for col in cat_cols:

            dist = df[col].value_counts(normalize=True)

            if dist.max() > 0.8:
                imbalance[col] = dist.to_dict()

        report["imbalance"] = imbalance

        return report

    # ---------------------------------------------------
    # FIX DATA
    # ---------------------------------------------------

    def fix(self, df: pd.DataFrame, strategy: dict):

        df_clean = df.copy()

        # Missing values
        if strategy.get("missing") == "median":
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

        elif strategy.get("missing") == "mean":
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

        elif strategy.get("missing") == "drop":
            df_clean = df_clean.dropna()

        # Duplicates
        if strategy.get("duplicates") == "drop":
            df_clean = df_clean.drop_duplicates()

        # Outliers
        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        if strategy.get("outliers") == "clip_iqr":

            for col in numeric_cols:

                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                df_clean[col] = df_clean[col].clip(lower, upper)

        elif strategy.get("outliers") == "remove":

            for col in numeric_cols:

                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                df_clean = df_clean[
                    (df_clean[col] >= lower) & (df_clean[col] <= upper)
                ]

        return df_clean

    # ---------------------------------------------------
    # COMPARE BEFORE / AFTER
    # ---------------------------------------------------

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame):

        comparison = pd.DataFrame({
            "metric": [
                "rows",
                "missing_values",
                "duplicates"
            ],
            "before": [
                len(df_before),
                df_before.isna().sum().sum(),
                df_before.duplicated().sum()
            ],
            "after": [
                len(df_after),
                df_after.isna().sum().sum(),
                df_after.duplicated().sum()
            ]
        })

        return comparison

    # ---------------------------------------------------
    # LLM EXPLANATION (ChatGPT API)
    # ---------------------------------------------------

    def explain_strategy(self, strategy: dict, report: dict):

        if self.client is None:
            raise ValueError("API key required for LLM explanation")

        prompt = f"""
Ты опытный специалист по Data Science.

Проанализируй отчет о качестве данных и объясни,
почему выбранная стратегия очистки данных является правильной.

Отвечай ТОЛЬКО на русском языке.

Найденные проблемы данных:
{report}

Выбранная стратегия очистки:
{strategy}

Дай краткое объяснение в терминах машинного обучения.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior data scientist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content


# ---------------------------------------------------
# OPTIONAL VISUALIZATION
# ---------------------------------------------------

def visualize_issues(df):

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Missing values
    plt.figure(figsize=(8,4))
    sns.heatmap(df.isnull(), cbar=False)
    plt.title("Missing Values")
    plt.show()

    # Outliers
    num_cols = df.select_dtypes(include="number").columns

    for col in num_cols:

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Outliers in {col}")
        plt.show()