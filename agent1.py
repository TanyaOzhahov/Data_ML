print("DATASET AGENT STARTED")

import os
import certifi
from dotenv import load_dotenv
from openai import OpenAI
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import HfApi

os.environ['SSL_CERT_FILE'] = certifi.where()

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------------------------
# Translate user query
# -------------------------

def translate_query(query):

    prompt = f"""
Translate the following query to English for dataset search.

Query:
{query}

Return only the translation.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return r.choices[0].message.content.strip()


# -------------------------
# LLM query generation
# -------------------------

def generate_search_queries(query):

    prompt = f"""
Generate 6 search queries to find machine learning datasets.

User request:
{query}

Focus on dataset discovery.

Return only queries, one per line.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )

    text = r.choices[0].message.content

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]

    return queries


# -------------------------
# Kaggle search
# -------------------------

def search_kaggle(query, max_results=5):

    api = KaggleApi()
    api.authenticate()

    datasets = api.dataset_list(search=query)

    results = []

    for d in datasets[:max_results]:

        results.append({
            "source": "Kaggle",
            "title": d.title,
            "description": d.subtitle,
            "url": f"https://www.kaggle.com/datasets/{d.ref}"
        })

    return results


# -------------------------
# HuggingFace search
# -------------------------

def search_huggingface(query, max_results=5):

    api = HfApi()

    datasets = list(api.list_datasets(search=query))

    results = []

    for d in datasets[:max_results]:

        desc = None

        if hasattr(d, "description"):
            desc = d.description

        if not desc:
            desc = "No description available"

        results.append({
            "source": "HuggingFace",
            "title": d.id,
            "description": desc,
            "url": f"https://huggingface.co/datasets/{d.id}"
        })

    return results


# -------------------------
# Remove duplicates
# -------------------------

def remove_duplicates(results):

    seen = set()
    unique = []

    for r in results:

        if r["url"] not in seen:

            seen.add(r["url"])
            unique.append(r)

    return unique


# -------------------------
# Filter text datasets
# -------------------------

def filter_text_datasets(results):

    keywords = ["text","corpus","nlp","document","language"]

    filtered = []

    for r in results:

        desc = (r["description"] or "").lower()

        if any(k in desc for k in keywords):

            filtered.append(r)

    return filtered


# -------------------------
# Format results
# -------------------------

def format_results(results):

    text = ""

    for r in results:

        text += f"""
Title: {r['title']}
Source: {r['source']}

Description:
{r['description']}

Link:
{r['url']}

------------------------------------
"""

    return text


# -------------------------
# Dataset agent
# -------------------------

def dataset_agent(user_query):

    print("\nTranslating query...")

    query_en = translate_query(user_query)

    print("Query:", query_en)

    print("\nGenerating search queries...")

    queries = generate_search_queries(query_en)

    for q in queries:
        print("Search:", q)

    results = []

    for q in queries:

        results += search_kaggle(q)
        results += search_huggingface(q)

    results = remove_duplicates(results)

    results = filter_text_datasets(results)

    results = results[:30]

    formatted = format_results(results)

    prompt = f"""
User request:
{user_query}

Datasets found:

{formatted}

Recommend the best datasets for the task.
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role":"user","content":prompt}]
    )

    recommendation = r.choices[0].message.content

    return formatted + "\nAI Recommendation:\n" + recommendation


# -------------------------
# MAIN
# -------------------------

def main():

    print("\nDataset Agent Ready\n")

    query = input("Введите запрос: ")

    result = dataset_agent(query)

    print("\nRESULTS\n")

    print(result)


if __name__ == "__main__":
    main()