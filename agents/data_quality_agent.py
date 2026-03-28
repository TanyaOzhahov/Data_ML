"""
SKILL: Data Quality Agent
━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : DataFrame (сырые данные)
OUTPUT: reports/data_quality_report.html  — интерактивный HTML-отчёт
        data/clean_dataset.csv            — очищенный датасет

Агент:
  1. detect_issues()      — находит пропуски, дубликаты, выбросы, дисбаланс
  2. fix()                — применяет стратегию очистки
  3. run_and_save_report()— генерирует красивый HTML-отчёт + сохраняет чистый CSV
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPORT_PATH = "reports/data_quality_report.html"
CLEAN_PATH = "data/clean_dataset.csv"


class DataQualityAgent:

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.model = model
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key) if key else None

    # ──────────────────────────────────────────────
    # 1. DETECT ISSUES
    # ──────────────────────────────────────────────

    def detect_issues(self, df: pd.DataFrame) -> dict:
        report: dict = {}

        # Missing values
        missing = df.isnull().sum()
        report["missing"] = missing[missing > 0].to_dict()

        # Duplicates
        report["duplicates"] = int(df.duplicated().sum())

        # Shape
        report["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}

        # Outliers (IQR)
        outliers: dict = {}
        for col in df.select_dtypes(include=np.number).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
            if mask.sum() > 0:
                outliers[col] = int(mask.sum())
        report["outliers"] = outliers

        # Class imbalance
        imbalance: dict = {}
        for col in df.select_dtypes(include="object").columns:
            dist = df[col].value_counts(normalize=True)
            if dist.max() > 0.8:
                imbalance[col] = {k: round(float(v), 3) for k, v in dist.items()}
        report["imbalance"] = imbalance

        return report

    # ──────────────────────────────────────────────
    # 2. FIX DATA
    # ──────────────────────────────────────────────

    def fix(self, df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
        df_clean = df.copy()

        missing_strat = strategy.get("missing", "median")
        if missing_strat == "median":
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
        elif missing_strat == "mean":
            df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))
        elif missing_strat == "drop":
            df_clean = df_clean.dropna()

        if strategy.get("duplicates") == "drop":
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        outlier_strat = strategy.get("outliers", "clip_iqr")
        for col in df_clean.select_dtypes(include=np.number).columns:
            q1, q3 = df_clean[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            if outlier_strat == "clip_iqr":
                df_clean[col] = df_clean[col].clip(lower, upper)
            elif outlier_strat == "remove":
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

        return df_clean

    # ──────────────────────────────────────────────
    # 3. COMPARE BEFORE / AFTER
    # ──────────────────────────────────────────────

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Metric": ["Rows", "Missing values", "Duplicate rows"],
            "Before": [
                len(df_before),
                int(df_before.isna().sum().sum()),
                int(df_before.duplicated().sum()),
            ],
            "After": [
                len(df_after),
                int(df_after.isna().sum().sum()),
                int(df_after.duplicated().sum()),
            ],
        })

    # ──────────────────────────────────────────────
    # 4. LLM EXPLANATION
    # ──────────────────────────────────────────────

    def explain_strategy(self, strategy: dict, report: dict) -> str:
        if self.client is None:
            return "LLM explanation skipped — OPENAI_API_KEY not set."
        prompt = (
            "Ты опытный Data Scientist. Кратко объясни (4-6 предложений), "
            "почему выбранная стратегия очистки данных подходит для ML задачи.\n\n"
            f"Найденные проблемы:\n{json.dumps(report, ensure_ascii=False, indent=2)}\n\n"
            f"Стратегия очистки:\n{json.dumps(strategy, ensure_ascii=False, indent=2)}"
        )
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return r.choices[0].message.content

    # ──────────────────────────────────────────────
    # 5. MAIN SKILL: RUN + SAVE HTML REPORT
    # ──────────────────────────────────────────────

    def run_and_save_report(
        self,
        df: pd.DataFrame,
        strategy: dict | None = None,
        report_path: str = REPORT_PATH,
        clean_path: str = CLEAN_PATH,
    ) -> pd.DataFrame:
        """
        Skill entry-point.
        Returns cleaned DataFrame; saves HTML report + clean CSV.
        """
        if strategy is None:
            strategy = {"missing": "median", "duplicates": "drop", "outliers": "clip_iqr"}

        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(clean_path) or ".", exist_ok=True)

        print("🔎 Detecting issues...")
        report = self.detect_issues(df)

        print("🧹 Cleaning data...")
        df_clean = self.fix(df, strategy)

        comparison = self.compare(df, df_clean)

        print("🤖 Generating LLM explanation...")
        explanation = self.explain_strategy(strategy, report)

        print("📄 Building HTML report...")
        html = self._build_html(df, df_clean, report, strategy, comparison, explanation)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ HTML report → {report_path}")

        df_clean.to_csv(clean_path, index=False, encoding="utf-8")
        print(f"✅ Clean CSV  → {clean_path}")

        return df_clean

    # ──────────────────────────────────────────────
    # HTML BUILDER
    # ──────────────────────────────────────────────

    def _build_html(self, df_before, df_after, report, strategy, comparison, explanation):
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Label distribution for text cols
        label_charts = ""
        for col in df_before.select_dtypes(include="object").columns[:2]:
            dist = df_before[col].value_counts().head(10)
            bars = "".join(
                f'<div class="bar-row"><span class="bar-label">{k}</span>'
                f'<div class="bar" style="width:{int(v/dist.max()*220)}px">{v}</div></div>'
                for k, v in dist.items()
            )
            label_charts += f'<div class="chart-box"><h3>Column: {col}</h3>{bars}</div>'

        # Comparison table rows
        cmp_rows = "".join(
            f"<tr><td>{r['Metric']}</td><td>{r['Before']}</td>"
            f"<td class='{'better' if r['After'] <= r['Before'] else 'worse'}'>{r['After']}</td></tr>"
            for _, r in comparison.iterrows()
        )

        # Missing values table
        missing_rows = "".join(
            f"<tr><td>{col}</td><td>{cnt}</td>"
            f"<td>{cnt/len(df_before)*100:.1f}%</td></tr>"
            for col, cnt in report["missing"].items()
        ) or "<tr><td colspan='3'>No missing values</td></tr>"

        # Outliers table
        outlier_rows = "".join(
            f"<tr><td>{col}</td><td>{cnt}</td></tr>"
            for col, cnt in report["outliers"].items()
        ) or "<tr><td colspan='2'>No outliers detected</td></tr>"

        # Strategy badges
        strategy_badges = "".join(
            f'<span class="badge">{k}: {v}</span>'
            for k, v in strategy.items()
        )

        # Sample of cleaned data
        sample_html = df_after.head(5).to_html(
            classes="data-table", border=0, index=False
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Quality Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #1a1a2e; }}
  .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             color: white; padding: 2rem; text-align: center; }}
  .header h1 {{ font-size: 2rem; margin-bottom: .3rem; }}
  .header p {{ opacity: .85; font-size: .9rem; }}
  .container {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ background: white; border-radius: 12px; padding: 1.5rem;
           margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .card h2 {{ font-size: 1.1rem; color: #667eea; margin-bottom: 1rem;
              border-bottom: 2px solid #f0f2f5; padding-bottom: .5rem; }}
  .card h3 {{ font-size: .95rem; color: #555; margin-bottom: .5rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
               gap: 1rem; margin-bottom: 1rem; }}
  .stat {{ background: #f7f8fc; border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat .val {{ font-size: 2rem; font-weight: 700; color: #667eea; }}
  .stat .lbl {{ font-size: .75rem; color: #888; margin-top: .2rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
  th {{ background: #667eea; color: white; padding: .6rem .8rem; text-align: left; }}
  td {{ padding: .55rem .8rem; border-bottom: 1px solid #f0f2f5; }}
  tr:hover td {{ background: #f7f8fc; }}
  .better {{ color: #27ae60; font-weight: 600; }}
  .worse  {{ color: #e74c3c; font-weight: 600; }}
  .badge {{ display: inline-block; background: #667eea; color: white;
            border-radius: 20px; padding: .2rem .7rem; font-size: .78rem; margin: .2rem; }}
  .bar-row {{ display: flex; align-items: center; margin: .3rem 0; }}
  .bar-label {{ width: 120px; font-size: .8rem; color: #555; flex-shrink: 0; }}
  .bar {{ background: linear-gradient(90deg, #667eea, #764ba2); height: 22px;
          border-radius: 4px; color: white; font-size: .75rem;
          display: flex; align-items: center; padding-left: .4rem; min-width: 30px; }}
  .chart-box {{ margin-bottom: 1rem; }}
  .explanation {{ background: #f7f8fc; border-left: 4px solid #667eea;
                  padding: 1rem; border-radius: 0 8px 8px 0; font-size: .9rem;
                  line-height: 1.6; white-space: pre-wrap; }}
  .data-table {{ font-size: .78rem; }}
  .footer {{ text-align: center; color: #aaa; font-size: .78rem; padding: 2rem; }}
  .tag {{ display: inline-block; background: #e8f4fd; color: #2980b9;
          border-radius: 4px; padding: .1rem .5rem; font-size: .78rem; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 Data Quality Report</h1>
  <p>Generated: {now}</p>
</div>

<div class="container">

  <!-- Overview -->
  <div class="card">
    <h2>📈 Overview</h2>
    <div class="stat-grid">
      <div class="stat"><div class="val">{report['shape']['rows']:,}</div><div class="lbl">Rows (before)</div></div>
      <div class="stat"><div class="val">{len(df_after):,}</div><div class="lbl">Rows (after)</div></div>
      <div class="stat"><div class="val">{report['shape']['cols']}</div><div class="lbl">Columns</div></div>
      <div class="stat"><div class="val">{sum(report['missing'].values())}</div><div class="lbl">Missing cells</div></div>
      <div class="stat"><div class="val">{report['duplicates']}</div><div class="lbl">Duplicates</div></div>
      <div class="stat"><div class="val">{sum(report['outliers'].values())}</div><div class="lbl">Outliers</div></div>
    </div>
  </div>

  <!-- Before / After -->
  <div class="card">
    <h2>🔄 Before vs After Cleaning</h2>
    <table>
      <thead><tr><th>Metric</th><th>Before</th><th>After</th></tr></thead>
      <tbody>{cmp_rows}</tbody>
    </table>
  </div>

  <!-- Strategy -->
  <div class="card">
    <h2>⚙️ Cleaning Strategy</h2>
    {strategy_badges}
  </div>

  <!-- Missing Values -->
  <div class="card">
    <h2>🕳️ Missing Values</h2>
    <table>
      <thead><tr><th>Column</th><th>Count</th><th>%</th></tr></thead>
      <tbody>{missing_rows}</tbody>
    </table>
  </div>

  <!-- Outliers -->
  <div class="card">
    <h2>📉 Outliers (IQR method)</h2>
    <table>
      <thead><tr><th>Column</th><th>Outlier count</th></tr></thead>
      <tbody>{outlier_rows}</tbody>
    </table>
  </div>

  <!-- Label Distribution -->
  <div class="card">
    <h2>📊 Category Distributions</h2>
    {label_charts if label_charts else "<p style='color:#aaa'>No categorical columns found.</p>"}
  </div>

  <!-- LLM Explanation -->
  <div class="card">
    <h2>🤖 AI Explanation</h2>
    <div class="explanation">{explanation}</div>
  </div>

  <!-- Data Sample -->
  <div class="card">
    <h2>🗂️ Cleaned Data Sample (first 5 rows)</h2>
    {sample_html}
  </div>

</div>
<div class="footer">Data Quality Agent · {now}</div>
</body>
</html>"""


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = input("Path to CSV (default: data/dataset.csv): ").strip() or "data/dataset.csv"
    df = pd.read_csv(csv_path)
    agent = DataQualityAgent()
    df_clean = agent.run_and_save_report(df)
    print(f"\nClean dataset shape: {df_clean.shape}")
