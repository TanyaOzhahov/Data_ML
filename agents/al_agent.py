"""
SKILL: Active Learning Agent  (Track A)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : labeled_df, pool_df, test_df (DataFrames)
OUTPUT: reports/learning_curve.png  — кривые обучения (entropy vs random)
        reports/al_report.json      — числовые результаты
        models/final_model.pkl      — финальная модель (joblib)

Track A — требования:
  • Старт: N = 50 размеченных примеров
  • 5 итераций AL-цикла
  • Сравнение стратегий: entropy vs random на одном графике
  • Вывод: сколько примеров экономит entropy vs random при одинаковом качестве
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt

from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


CURVE_PATH  = "reports/learning_curve.png"
REPORT_PATH = "reports/al_report.json"
MODEL_PATH  = "models/final_model.pkl"


class ActiveLearningAgent:

    def __init__(self, model_type: str = "logreg"):
        self.model_type = model_type
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        self.vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True)
        self._fitted_vectorizer = False

    # ──────────────────────────────────────────────
    # FIT (train on labeled data)
    # ──────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, text_col: str, label_col: str):
        if not self._fitted_vectorizer:
            X = self.vectorizer.fit_transform(df[text_col].astype(str))
            self._fitted_vectorizer = True
        else:
            X = self.vectorizer.transform(df[text_col].astype(str))
        self.model.fit(X, df[label_col])

    # ──────────────────────────────────────────────
    # QUERY STRATEGY
    # ──────────────────────────────────────────────

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = "entropy",
        batch_size: int = 10,
        text_col: str = "text",
    ) -> np.ndarray:
        """
        Returns indices of the most informative samples from pool_df.
        Strategies: 'entropy', 'margin', 'random'
        """
        if pool_df.empty:
            return np.array([], dtype=int)

        actual_batch = min(batch_size, len(pool_df))
        X_pool = self.vectorizer.transform(pool_df[text_col].astype(str))
        probs = self.model.predict_proba(X_pool)

        if strategy == "entropy":
            scores = scipy_entropy(probs.T)          # higher = more uncertain

        elif strategy == "margin":
            sorted_p = -np.sort(-probs, axis=1)
            scores = -(sorted_p[:, 0] - sorted_p[:, 1])   # lower margin = higher score

        elif strategy == "random":
            scores = np.random.rand(len(pool_df))

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        return np.argsort(scores)[-actual_batch:]

    # ──────────────────────────────────────────────
    # EVALUATE
    # ──────────────────────────────────────────────

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str,
        label_col: str,
    ) -> dict:
        X_train = self.vectorizer.transform(train_df[text_col].astype(str))
        X_test  = self.vectorizer.transform(test_df[text_col].astype(str))

        self.model.fit(X_train, train_df[label_col])
        preds = self.model.predict(X_test)

        return {
            "accuracy": float(accuracy_score(test_df[label_col], preds)),
            "f1":       float(f1_score(test_df[label_col], preds, average="macro")),
        }

    # ──────────────────────────────────────────────
    # RUN ONE AL CYCLE
    # ──────────────────────────────────────────────

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 10,
        text_col: str = "text",
        label_col: str = "label",
    ) -> list[dict]:
        """
        Runs one complete AL cycle for a given strategy.
        Returns history list: [{iteration, n_labeled, accuracy, f1}, ...]
        """
        # Reset model state for fair comparison
        self.model = LogisticRegression(max_iter=1000, C=1.0)

        history = []
        labeled = labeled_df.copy()
        pool    = pool_df.copy()

        for i in range(n_iterations):
            if pool.empty:
                print(f"  ⚠️  Pool empty at iteration {i} — stopping early")
                break

            # Fit vectorizer on first iteration only (shared across strategies)
            if i == 0 and not self._fitted_vectorizer:
                self.vectorizer.fit_transform(
                    pd.concat([labeled, pool, test_df])[text_col].astype(str)
                )
                self._fitted_vectorizer = True

            metrics = self.evaluate(labeled, test_df, text_col, label_col)

            history.append({
                "iteration": i,
                "n_labeled": len(labeled),
                "accuracy":  metrics["accuracy"],
                "f1":        metrics["f1"],
                "strategy":  strategy,
            })

            print(
                f"  [{strategy}] Iter {i+1}/{n_iterations} | "
                f"Labeled: {len(labeled):4d} | Pool: {len(pool):4d} | "
                f"Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}"
            )

            idx = self.query(pool, strategy=strategy, batch_size=batch_size, text_col=text_col)
            if len(idx) == 0:
                print("  ⚠️  Query returned 0 indices — stopping")
                break

            labeled = pd.concat([labeled, pool.iloc[idx]], ignore_index=True)
            pool    = pool.drop(pool.index[idx]).reset_index(drop=True)

        return history

    # ──────────────────────────────────────────────
    # SAMPLE SAVINGS ANALYSIS
    # ──────────────────────────────────────────────

    @staticmethod
    def compute_sample_savings(
        history_entropy: list[dict],
        history_random: list[dict],
        target_accuracy: float | None = None,
    ) -> dict:
        """
        Finds target accuracy (default: max of random baseline)
        and computes how many fewer labeled samples entropy needs to reach it.
        """
        df_e = pd.DataFrame(history_entropy)
        df_r = pd.DataFrame(history_random)

        if target_accuracy is None:
            target_accuracy = df_r["accuracy"].max() * 0.98   # 98% of random's best

        # First iteration where each strategy reaches the target
        def first_reach(df, target):
            hit = df[df["accuracy"] >= target]
            if hit.empty:
                return None
            return int(hit.iloc[0]["n_labeled"])

        n_entropy = first_reach(df_e, target_accuracy)
        n_random  = first_reach(df_r, target_accuracy)

        if n_entropy is not None and n_random is not None:
            saved = n_random - n_entropy
            pct   = saved / n_random * 100 if n_random > 0 else 0
        else:
            saved = None
            pct   = None

        return {
            "target_accuracy": round(target_accuracy, 4),
            "n_labeled_entropy": n_entropy,
            "n_labeled_random":  n_random,
            "samples_saved":     saved,
            "savings_pct":       round(pct, 1) if pct is not None else None,
        }

    # ──────────────────────────────────────────────
    # PLOT LEARNING CURVES
    # ──────────────────────────────────────────────

    def plot_curves(
        self,
        histories: dict[str, list[dict]],
        output_path: str = CURVE_PATH,
        savings: dict | None = None,
    ):
        """
        Plots learning curves for all strategies on one figure.
        histories = {"entropy": [...], "random": [...]}
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Active Learning — Learning Curves", fontsize=14, fontweight="bold")

        colors   = {"entropy": "#667eea", "random": "#f39c12", "margin": "#27ae60"}
        lstyles  = {"entropy": "-o",      "random": "--s",     "margin": "-^"}

        for metric, ax, title in zip(
            ["accuracy", "f1"],
            axes,
            ["Accuracy", "F1-score (macro)"],
        ):
            for strategy, history in histories.items():
                df_h = pd.DataFrame(history)
                color = colors.get(strategy, "gray")
                style = lstyles.get(strategy, "-o")
                ax.plot(
                    df_h["n_labeled"],
                    df_h[metric],
                    style,
                    color=color,
                    label=strategy.capitalize(),
                    linewidth=2,
                    markersize=6,
                )

            if savings and metric == "accuracy" and savings.get("target_accuracy"):
                ax.axhline(
                    y=savings["target_accuracy"],
                    color="red",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Target ({savings['target_accuracy']:.3f})",
                )
                # Annotate vertical lines
                for key, strat in [("n_labeled_entropy", "entropy"), ("n_labeled_random", "random")]:
                    n = savings.get(key)
                    if n:
                        ax.axvline(
                            x=n,
                            color=colors.get(strat, "gray"),
                            linestyle=":",
                            linewidth=1.2,
                            alpha=0.6,
                        )

            ax.set_xlabel("Number of labeled samples", fontsize=11)
            ax.set_ylabel(title, fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.05)

        # Savings annotation
        if savings and savings.get("samples_saved") is not None:
            ann = (
                f"📊 Savings: entropy reaches target accuracy\n"
                f"with {savings['n_labeled_entropy']} samples vs "
                f"{savings['n_labeled_random']} (random)\n"
                f"→ {savings['samples_saved']} samples saved "
                f"({savings['savings_pct']}%)"
            )
            fig.text(
                0.5, -0.04, ann,
                ha="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f7f8fc", ec="#667eea", alpha=0.9),
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Learning curve → {output_path}")

    # ──────────────────────────────────────────────
    # MAIN SKILL: COMPARE STRATEGIES
    # ──────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        n_start: int = 50,
        n_iterations: int = 5,
        batch_size: int = 10,
        strategies: list[str] | None = None,
    ) -> dict:
        """
        Skill entry-point.

        Splits data into labeled (n_start), pool, test.
        Runs AL cycle for each strategy.
        Saves: learning_curve.png, al_report.json, final_model.pkl
        Returns full results dict.
        """
        os.makedirs("reports", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        if strategies is None:
            strategies = ["entropy", "random"]

        print(f"\n📐 Dataset: {len(df)} rows | text_col={text_col!r} | label_col={label_col!r}")

        # Shuffle + split
        df = df.dropna(subset=[text_col, label_col]).sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        n_test = max(50, int(0.2 * n))

        test_df    = df.iloc[-n_test:]
        remaining  = df.iloc[:-n_test]

        n_start    = min(n_start, int(0.5 * len(remaining)))
        labeled_df = remaining.iloc[:n_start]
        pool_df    = remaining.iloc[n_start:]

        print(f"  Start labeled: {len(labeled_df)} | Pool: {len(pool_df)} | Test: {len(test_df)}")

        # Pre-fit vectorizer on all text (shared across strategies)
        print("\n🔤 Fitting vectorizer...")
        all_text = pd.concat([labeled_df, pool_df, test_df])[text_col].astype(str)
        self.vectorizer.fit(all_text)
        self._fitted_vectorizer = True

        # Run each strategy
        all_histories: dict[str, list[dict]] = {}

        for strategy in strategies:
            print(f"\n{'='*50}")
            print(f"▶  Strategy: {strategy.upper()}")
            print("=" * 50)

            # Fresh model per strategy
            self.model = LogisticRegression(max_iter=1000, C=1.0)

            history = self.run_cycle(
                labeled_df=labeled_df.copy(),
                pool_df=pool_df.copy(),
                test_df=test_df,
                strategy=strategy,
                n_iterations=n_iterations,
                batch_size=batch_size,
                text_col=text_col,
                label_col=label_col,
            )
            all_histories[strategy] = history

        # Sample savings analysis
        savings = {}
        if "entropy" in all_histories and "random" in all_histories:
            savings = self.compute_sample_savings(
                all_histories["entropy"],
                all_histories["random"],
            )
            print(f"\n💡 Sample savings: {savings}")

        # Plot
        self.plot_curves(all_histories, output_path=CURVE_PATH, savings=savings)

        # Final report
        report = {
            "config": {
                "n_start":      n_start,
                "n_iterations": n_iterations,
                "batch_size":   batch_size,
                "strategies":   strategies,
                "test_size":    len(test_df),
            },
            "histories": all_histories,
            "savings":   savings,
        }

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"✅ Report JSON → {REPORT_PATH}")

        # Save final model (entropy strategy if available, else first)
        best_strategy = "entropy" if "entropy" in strategies else strategies[0]
        self.model = LogisticRegression(max_iter=1000, C=1.0)
        X_final = self.vectorizer.transform(df[text_col].astype(str))
        self.model.fit(X_final, df[label_col])

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "model": self.model}, f)
        print(f"✅ Model saved → {MODEL_PATH}")

        return report


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    csv_path  = input("Path to CSV (default: data/labeled_dataset.csv): ").strip() or "data/labeled_dataset.csv"
    text_col  = input("Text column (default: text): ").strip() or "text"
    label_col = input("Label column (default: label_auto): ").strip() or "label_auto"

    df = pd.read_csv(csv_path)
    agent = ActiveLearningAgent()
    results = agent.run(df, text_col=text_col, label_col=label_col)

    savings = results.get("savings", {})
    if savings.get("samples_saved") is not None:
        print(f"\n🏁 Result: entropy saves {savings['samples_saved']} samples "
              f"({savings['savings_pct']}%) vs random at accuracy={savings['target_accuracy']:.3f}")
