"""
SKILL: Dataset Collection Agent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT : user query (str)
OUTPUT: data/raw_datasets.csv  — каталог найденных датасетов
        data/dataset.csv       — финальный датасет для обучения (если удалось скачать)

Агент:
  1. Переводит запрос на английский
  2. Генерирует 6 поисковых запросов через LLM
  3. Ищет на Kaggle и HuggingFace
  4. Дедуплицирует + фильтрует текстовые датасеты
  5. Сохраняет каталог в data/raw_datasets.csv
  6. Пытается скачать лучший HuggingFace датасет в data/dataset.csv
"""

import os
import certifi
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import HfApi

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

CATALOG_PATH = "data/raw_datasets.csv"
DATASET_PATH = "data/dataset.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def translate_query(query: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": f"Translate to English for dataset search. Return only translation.\n\n{query}"
        }]
    )
    return r.choices[0].message.content.strip()


def generate_search_queries(query: str) -> list[str]:
    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate 6 diverse search queries to find machine learning datasets for:\n{query}\n"
                "Return only queries, one per line, no numbering or bullets."
            )
        }]
    )
    return [q.strip("- ").strip() for q in r.choices[0].message.content.splitlines() if q.strip()]


def search_kaggle(query: str, max_results: int = 5) -> list[dict]:
    try:
        api = KaggleApi()
        api.authenticate()
        datasets = api.dataset_list(search=query)
        results = []
        for d in datasets[:max_results]:
            results.append({
                "source": "Kaggle",
                "title": d.title,
                "description": d.subtitle or "",
                "url": f"https://www.kaggle.com/datasets/{d.ref}",
                "ref": d.ref,
                "size_mb": "",
            })
        return results
    except Exception as e:
        print(f"  ⚠️  Kaggle error: {e}")
        return []


def search_huggingface(query: str, max_results: int = 5) -> list[dict]:
    try:
        api = HfApi()
        datasets = list(api.list_datasets(search=query, limit=max_results))
        results = []
        for d in datasets:
            results.append({
                "source": "HuggingFace",
                "title": d.id,
                "description": getattr(d, "description", "") or "",
                "url": f"https://huggingface.co/datasets/{d.id}",
                "ref": d.id,
                "size_mb": "",
            })
        return results
    except Exception as e:
        print(f"  ⚠️  HuggingFace error: {e}")
        return []


def is_text_dataset(row: dict) -> bool:
    text = ((row.get("description") or "") + " " + (row.get("title") or "")).lower()
    keywords = ["text", "corpus", "nlp", "document", "language",
                "sentiment", "review", "tweet", "opinion", "classification"]
    return any(k in text for k in keywords)


def pick_best_hf_dataset(catalog: pd.DataFrame) -> str | None:
    """Returns ref of the best HuggingFace dataset for download."""
    hf = catalog[catalog["source"] == "HuggingFace"]
    if hf.empty:
        return None
    # prefer datasets with 'sentiment' or 'review' in title
    pref = hf[hf["title"].str.contains("sentiment|review|imdb", case=False, na=False)]
    if not pref.empty:
        return pref.iloc[0]["ref"]
    return hf.iloc[0]["ref"]


def download_hf_dataset(ref: str, output_path: str) -> bool:
    """Try to download first split of an HF dataset into CSV."""
    try:
        from datasets import load_dataset  # type: ignore
        print(f"  Downloading HuggingFace dataset: {ref} ...")
        ds = load_dataset(ref, split="train", trust_remote_code=True)
        df = ds.to_pandas()
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"  ✅ Downloaded {len(df)} rows → {output_path}")
        return True
    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")
        return False


# ──────────────────────────────────────────────
# MAIN SKILL
# ──────────────────────────────────────────────

def run(user_query: str) -> pd.DataFrame:
    """
    Skill entry-point.
    Returns catalog DataFrame; side-effects: saves CSV files.
    """
    os.makedirs("data", exist_ok=True)

    print("🔤 Translating query...")
    query_en = translate_query(user_query)
    print(f"   → {query_en}")

    print("🔍 Generating search queries...")
    queries = generate_search_queries(query_en)
    for q in queries:
        print(f"   • {q}")

    all_results: list[dict] = []
    for q in queries:
        print(f"\n📡 Searching: {q}")
        all_results += search_kaggle(q)
        all_results += search_huggingface(q)

    # Deduplicate by URL
    seen: set[str] = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    # Filter text datasets
    filtered = [r for r in unique if is_text_dataset(r)]

    catalog = pd.DataFrame(filtered[:50])

    # ── Save catalog CSV ──
    catalog.to_csv(CATALOG_PATH, index=False, encoding="utf-8")
    print(f"\n✅ Catalog saved: {CATALOG_PATH}  ({len(catalog)} datasets)")

    # ── Try to download best dataset ──
    best_ref = pick_best_hf_dataset(catalog)
    if best_ref:
        print(f"\n⬇️  Attempting to download: {best_ref}")
        success = download_hf_dataset(best_ref, DATASET_PATH)
        if not success:
            print("   Falling back to local train.csv if present...")
            if os.path.exists("train.csv"):
                import shutil
                shutil.copy("train.csv", DATASET_PATH)
                print(f"   ✅ Copied train.csv → {DATASET_PATH}")
    else:
        print("⚠️  No suitable HF dataset found; using local train.csv")
        if os.path.exists("train.csv"):
            import shutil
            shutil.copy("train.csv", DATASET_PATH)

    return catalog


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    query = input("Введите запрос: ")
    catalog = run(query)
    print("\nTop results:")
    print(catalog[["source", "title", "url"]].to_string(index=False))
