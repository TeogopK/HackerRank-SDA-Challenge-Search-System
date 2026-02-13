import re
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------
# Configuration
# ----------------------------
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CHALLENGES_DIR = str(PROJECT_ROOT / "scraped_data" / "challenges")
LABELS_DIR = str(PROJECT_ROOT / "openai_annotations")


# ----------------------------
# Text preprocessing
# ----------------------------
SECTION_PATTERN = re.compile(
    r"(PROBLEM STATEMENT:|INPUT FORMAT:|OUTPUT FORMAT:|CONSTRAINTS:)"
)


def extract_sections(text: str) -> str:
    """Extract only problem statement, input format, and output format.
    Drops: metadata header (name, slug, difficulty, score) and constraints."""
    parts = SECTION_PATTERN.split(text)

    # parts looks like: [header, "PROBLEM STATEMENT:", content, "INPUT FORMAT:", content, ...]
    sections = {}
    for i, part in enumerate(parts):
        part_stripped = part.strip().rstrip(":")
        if part_stripped in ("PROBLEM STATEMENT", "INPUT FORMAT", "OUTPUT FORMAT"):
            if i + 1 < len(parts):
                sections[part_stripped] = parts[i + 1]

    kept = []
    for key in ("PROBLEM STATEMENT", "INPUT FORMAT", "OUTPUT FORMAT"):
        if key in sections:
            kept.append(sections[key])

    return " ".join(kept) if kept else text


def normalize(text: str) -> str:
    if not text:
        return ""
    text = extract_sections(text)
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)   # markdown images
    text = re.sub(r"\$[^$]*\$", " ", text)         # LaTeX math
    text = re.sub(r"\*+", " ", text)               # bold/italic markers
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)  # keep letters + digits
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------
# Load challenges with labels
# ----------------------------
def load_challenges() -> List[Dict]:
    docs = []

    for txt_path in Path(CHALLENGES_DIR).glob("*.txt"):
        label_path = Path(LABELS_DIR) / f"{txt_path.stem}.json"
        if not label_path.exists():
            continue

        raw_text = txt_path.read_text(encoding="utf-8")
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        ann = meta.get("annotation", {})

        docs.append(
            {
                "slug": txt_path.stem,
                "name": txt_path.stem,
                "raw_text": raw_text,
                "text": normalize(raw_text),
                "primary_topic": ann.get("primary_topic", "unknown"),
            }
        )

    if not docs:
        raise RuntimeError("No challenge TXT files found with labels.")

    return docs


# ----------------------------
# Build BERT embeddings index
# ----------------------------
def build_index(docs: List[Dict], model: SentenceTransformer):
    corpus = [d["text"] for d in docs]
    print(f"Encoding {len(corpus)} documents with {MODEL_NAME}...")
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=32)
    return embeddings


# ----------------------------
# Similarity search
# ----------------------------
def find_similar(
    query_vec,
    embeddings,
    docs: List[Dict],
    exclude_idx: int = -1,
    top_k: int = 5,
):
    similarities = cosine_similarity([query_vec], embeddings)[0]

    if exclude_idx >= 0:
        similarities[exclude_idx] = -1.0

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "slug": docs[i]["slug"],
            "name": docs[i]["name"],
            "score": float(similarities[i]),
            "primary_topic": docs[i]["primary_topic"],
        }
        for i in top_indices
    ]


# ----------------------------
# Retrieval metrics
# ----------------------------
def precision_at_k(results: List[Dict], query_topic: str, k: int) -> float:
    top = results[:k]
    relevant = sum(1 for r in top if r["primary_topic"] == query_topic)
    return relevant / k


def reciprocal_rank(results: List[Dict], query_topic: str) -> float:
    for i, r in enumerate(results, 1):
        if r["primary_topic"] == query_topic:
            return 1.0 / i
    return 0.0


def average_precision(results: List[Dict], query_topic: str, k: int) -> float:
    hits = 0
    sum_precision = 0.0
    for i, r in enumerate(results[:k], 1):
        if r["primary_topic"] == query_topic:
            hits += 1
            sum_precision += hits / i
    return sum_precision / hits if hits else 0.0


def evaluate_retrieval(docs: List[Dict], embeddings, top_k: int = 5):
    all_results = []
    p1_scores, p3_scores, p5_scores = [], [], []
    rr_scores = []
    ap_scores = []

    for idx, doc in enumerate(docs):
        query_vec = embeddings[idx]
        results = find_similar(query_vec, embeddings, docs, exclude_idx=idx, top_k=top_k)
        topic = doc["primary_topic"]

        all_results.append(
            {
                "query_slug": doc["slug"],
                "query_topic": topic,
                "results": results,
            }
        )

        p1_scores.append(precision_at_k(results, topic, 1))
        p3_scores.append(precision_at_k(results, topic, min(3, len(results))))
        p5_scores.append(precision_at_k(results, topic, min(5, len(results))))
        rr_scores.append(reciprocal_rank(results, topic))
        ap_scores.append(average_precision(results, topic, top_k))

    n = len(docs)
    metrics = {
        "model": MODEL_NAME,
        "P@1": round(sum(p1_scores) / n, 3),
        "P@3": round(sum(p3_scores) / n, 3),
        "P@5": round(sum(p5_scores) / n, 3),
        "MRR": round(sum(rr_scores) / n, 3),
        "MAP@5": round(sum(ap_scores) / n, 3),
        "total_queries": n,
    }

    # Per-topic breakdown
    topic_groups = defaultdict(list)
    for s, p1 in zip(docs, p1_scores):
        topic_groups[s["primary_topic"]].append(p1)

    per_topic = {}
    for topic, scores in sorted(topic_groups.items()):
        per_topic[topic] = {
            "count": len(scores),
            "P@1": round(sum(scores) / len(scores), 3),
        }
    metrics["per_topic"] = per_topic

    return all_results, metrics


# ----------------------------
# Query from file
# ----------------------------
def query_from_file(
    query_path: str,
    docs: List[Dict],
    model: SentenceTransformer,
    embeddings,
    top_k: int = 5,
):
    raw = Path(query_path).read_text(encoding="utf-8")
    query_vec = model.encode(normalize(raw))
    similarities = cosine_similarity([query_vec], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [
        {
            "slug": docs[i]["slug"],
            "name": docs[i]["name"],
            "score": float(similarities[i]),
            "primary_topic": docs[i]["primary_topic"],
        }
        for i in top_indices
    ]


# ----------------------------
# Main
# ----------------------------
def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading scraped challenges...")
    docs = load_challenges()
    print(f"Loaded {len(docs)} challenges with labels")

    embeddings = build_index(docs, model)

    print("Evaluating leave-one-out retrieval...")
    all_results, metrics = evaluate_retrieval(docs, embeddings, top_k=5)

    # Save results
    results_dir = PROJECT_ROOT / "results" / "bert"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "bert_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    with open(results_dir / "bert_similarity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print("  results/bert/bert_similarity_results.json")
    print("  results/bert/bert_similarity_metrics.json")

    print(f"\nRetrieval metrics (BERT: {MODEL_NAME}):")
    for k, v in metrics.items():
        if k in ("per_topic", "model"):
            continue
        print(f"  {k}: {v}")

    print("\nPer-topic P@1:")
    for topic, info in metrics["per_topic"].items():
        print(f"  {topic} (n={info['count']}): P@1={info['P@1']}")

    # Optional: query from file
    query_path = PROJECT_ROOT / "query_challenge.txt"
    if query_path.exists():
        print("\n--- Ad-hoc query from query_challenge.txt ---")
        results = query_from_file(str(query_path), docs, model, embeddings, top_k=5)
        for rank, r in enumerate(results, 1):
            print(f"  {rank}. {r['slug']}  (score: {r['score']:.3f}, topic: {r['primary_topic']})")


if __name__ == "__main__":
    main()
