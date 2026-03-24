import pandas as pd
from al_agent import ActiveLearningAgent


TEXT_COL = "review"
LABEL_COL = "sentiment"
# загружаем данные
df = pd.read_csv("train.csv")

# делим
df = df.sample(frac=1, random_state=42)

labeled_df = df.iloc[:50]
pool_df = df.iloc[50:1000]
test_df = df.iloc[1000:]

agent = ActiveLearningAgent()

# ENTROPY стратегия
history_entropy = agent.run_cycle(
    labeled_df.copy(),
    pool_df.copy(),
    test_df,
    strategy="entropy",
    text_col=TEXT_COL,
    label_col=LABEL_COL
)

# RANDOM стратегия
history_random = agent.run_cycle(
    labeled_df.copy(),
    pool_df.copy(),
    test_df,
    strategy="random",
    text_col=TEXT_COL,
    label_col=LABEL_COL
)

# график
agent.report(history_entropy, label="entropy")
agent.report(history_random, label="random")