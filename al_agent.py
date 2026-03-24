import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
import matplotlib.pyplot as plt


class ActiveLearningAgent:

    def __init__(self, model='logreg'):
        self.model_name = model
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = TfidfVectorizer(max_features=5000)

    # ---------------------------------------------------
    # FIT MODEL
    # ---------------------------------------------------

    def fit(self, df, text_col="review", label_col="sentiment"):

        X = self.vectorizer.fit_transform(df[text_col])
        y = df[label_col]

        self.model.fit(X, y)

        return self.model

    # ---------------------------------------------------
    # QUERY STRATEGY
    # ---------------------------------------------------

    def query(self, pool_df, strategy="entropy", batch_size=10, text_col="review"):

        X_pool = self.vectorizer.transform(pool_df[text_col])
        probs = self.model.predict_proba(X_pool)

        if strategy == "entropy":
            scores = entropy(probs.T)

        elif strategy == "margin":
            sorted_probs = -np.sort(-probs, axis=1)
            scores = sorted_probs[:, 0] - sorted_probs[:, 1]
            scores = -scores  # чем меньше margin — тем более uncertain

        elif strategy == "random":
            scores = np.random.rand(len(pool_df))

        else:
            raise ValueError("Unknown strategy")

        indices = np.argsort(scores)[-batch_size:]

        return indices

    # ---------------------------------------------------
    # EVALUATE
    # ---------------------------------------------------

    def evaluate(self, train_df, test_df, text_col="review", label_col="sentiment"):

        X_train = self.vectorizer.transform(train_df[text_col])
        y_train = train_df[label_col]

        X_test = self.vectorizer.transform(test_df[text_col])
        y_test = test_df[label_col]

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="macro")
        }

    # ---------------------------------------------------
    # RUN ACTIVE LEARNING LOOP
    # ---------------------------------------------------

    def run_cycle(
        self,
        labeled_df,
        pool_df,
        test_df,
        strategy="entropy",
        n_iterations=5,
        batch_size=20,
        text_col="text",
        label_col="label"
    ):

        history = []

        for i in range(n_iterations):

            # обучаемся
            self.fit(labeled_df, text_col, label_col)

            # оцениваем
            metrics = self.evaluate(labeled_df, test_df, text_col, label_col)

            history.append({
                "iteration": i,
                "n_labeled": len(labeled_df),
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"]
            })

            # выбираем новые данные
            idx = self.query(pool_df, strategy, batch_size, text_col)

            new_samples = pool_df.iloc[idx]

            # добавляем в labeled
            labeled_df = pd.concat([labeled_df, new_samples])

            # удаляем из pool
            pool_df = pool_df.drop(pool_df.index[idx])

        return history

    # ---------------------------------------------------
    # REPORT (LEARNING CURVE)
    # ---------------------------------------------------

    def report(self, history, label="strategy"):

        df_hist = pd.DataFrame(history)

        plt.plot(df_hist["n_labeled"], df_hist["accuracy"], label=f"{label}-accuracy")
        plt.plot(df_hist["n_labeled"], df_hist["f1"], label=f"{label}-f1")

        plt.xlabel("Number of labeled samples")
        plt.ylabel("Score")
        plt.title("Active Learning Curve")
        plt.legend()

        plt.savefig("learning_curve.png")
        plt.show()