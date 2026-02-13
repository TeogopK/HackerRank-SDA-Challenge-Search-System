import re
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
import classla
import stopwordsiso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------
# Configuration
# ----------------------------
BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BERT_WEIGHT = 0.4  # tune this: 0.0 = pure TF-IDF, 1.0 = pure BERT

CHALLENGES_DIR = str(PROJECT_ROOT / "scraped_data" / "challenges")
LABELS_DIR = str(PROJECT_ROOT / "openai_annotations")

# ----------------------------
# NLP pipeline (for TF-IDF branch)
# ----------------------------
nlp = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)
STOPWORDS = set(stopwordsiso.stopwords("bg"))

# ----------------------------
# Text preprocessing
# ----------------------------
SECTION_PATTERN = re.compile(
    r"(PROBLEM STATEMENT:|INPUT FORMAT:|OUTPUT FORMAT:|CONSTRAINTS:)"
)


def extract_sections(text: str) -> str:
    parts = SECTION_PATTERN.split(text)
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


def clean_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\*+", " ", text)
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_bert(text: str) -> str:
    if not text:
        return ""
    return clean_markup(extract_sections(text))


def lemmatize(text: str) -> List[str]:
    doc = nlp(text)
    lemmas = []
    for sent in doc.sentences:
        for w in sent.words:
            if not w.lemma or w.lemma in STOPWORDS or not w.lemma.isalpha():
                continue
            lemmas.append(w.lemma)
    return lemmas


def preprocess_tfidf(text: str) -> str:
    if not text:
        return ""
    text = extract_sections(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(lemmatize(text))


# ----------------------------
# Load challenges
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
                "primary_topic": ann.get("primary_topic", "unknown"),
            }
        )
    if not docs:
        raise RuntimeError("No challenge TXT files found with labels.")
    return docs


# ----------------------------
# Build both indexes
# ----------------------------
def build_tfidf_index(docs: List[Dict]):
    print("Preprocessing for TF-IDF (lemmatization)...")
    corpus = []
    for i, d in enumerate(docs):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(docs)}")
        corpus.append(preprocess_tfidf(d["raw_text"]))

    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def build_bert_index(docs: List[Dict], model: SentenceTransformer):
    corpus = [preprocess_bert(d["raw_text"]) for d in docs]
    print(f"Encoding {len(corpus)} documents with {BERT_MODEL}...")
    embeddings = model.encode(corpus, show_progress_bar=True, batch_size=32)
    return embeddings


# ----------------------------
# Hybrid similarity
# ----------------------------
def hybrid_similarities(tfidf_matrix, bert_embeddings, idx: int, alpha: float):
    tfidf_sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
    bert_sims = cosine_similarity([bert_embeddings[idx]], bert_embeddings)[0]
    return (1 - alpha) * tfidf_sims + alpha * bert_sims


def find_similar(
    idx: int,
    tfidf_matrix,
    bert_embeddings,
    docs: List[Dict],
    alpha: float,
    top_k: int = 5,
):
    similarities = hybrid_similarities(tfidf_matrix, bert_embeddings, idx, alpha)
    similarities[idx] = -1.0

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
# Metrics
# ----------------------------
def precision_at_k(results: List[Dict], query_topic: str, k: int) -> float:
    return sum(1 for r in results[:k] if r["primary_topic"] == query_topic) / k


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


def evaluate(docs, tfidf_matrix, bert_embeddings, alpha, top_k=5):
    all_results = []
    p1, p3, p5, rr, ap = [], [], [], [], []

    for idx, doc in enumerate(docs):
        results = find_similar(idx, tfidf_matrix, bert_embeddings, docs, alpha, top_k)
        topic = doc["primary_topic"]
        all_results.append({"query_slug": doc["slug"], "query_topic": topic, "results": results})
        p1.append(precision_at_k(results, topic, 1))
        p3.append(precision_at_k(results, topic, min(3, len(results))))
        p5.append(precision_at_k(results, topic, min(5, len(results))))
        rr.append(reciprocal_rank(results, topic))
        ap.append(average_precision(results, topic, top_k))

    n = len(docs)
    metrics = {
        "model": f"hybrid (BERT_WEIGHT={alpha})",
        "P@1": round(sum(p1) / n, 3),
        "P@3": round(sum(p3) / n, 3),
        "P@5": round(sum(p5) / n, 3),
        "MRR": round(sum(rr) / n, 3),
        "MAP@5": round(sum(ap) / n, 3),
        "total_queries": n,
    }

    topic_groups = defaultdict(list)
    for s, score in zip(docs, p1):
        topic_groups[s["primary_topic"]].append(score)
    metrics["per_topic"] = {
        t: {"count": len(sc), "P@1": round(sum(sc) / len(sc), 3)}
        for t, sc in sorted(topic_groups.items())
    }

    return all_results, metrics


# ----------------------------
# Main
# ----------------------------
def main():
    print("Loading challenges...")
    docs = load_challenges()
    print(f"Loaded {len(docs)} challenges")

    vectorizer, tfidf_matrix = build_tfidf_index(docs)

    model = SentenceTransformer(BERT_MODEL)
    bert_embeddings = build_bert_index(docs, model)

    # Sweep weights to find the best blend
    print("\n--- Weight sweep ---")
    best_alpha, best_p1 = 0.0, 0.0
    for alpha_int in range(0, 11, 1):
        alpha = alpha_int / 10.0
        _, m = evaluate(docs, tfidf_matrix, bert_embeddings, alpha, top_k=5)
        tag = ""
        if m["P@1"] > best_p1:
            best_p1 = m["P@1"]
            best_alpha = alpha
            tag = " <-- best"
        print(f"  alpha={alpha:.1f}  P@1={m['P@1']}  MRR={m['MRR']}  MAP@5={m['MAP@5']}{tag}")

    print(f"\nBest alpha: {best_alpha:.1f} (P@1={best_p1})")

    # Full eval at best weight
    all_results, metrics = evaluate(docs, tfidf_matrix, bert_embeddings, best_alpha, top_k=5)

    results_dir = PROJECT_ROOT / "results" / "hybrid"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "hybrid_similarity_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(results_dir / "hybrid_similarity_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print("  results/hybrid/hybrid_similarity_results.json")
    print("  results/hybrid/hybrid_similarity_metrics.json")

    print(f"\nRetrieval metrics (hybrid, alpha={best_alpha:.1f}):")
    for k, v in metrics.items():
        if k in ("per_topic", "model"):
            continue
        print(f"  {k}: {v}")

    print("\nPer-topic P@1:")
    for topic, info in metrics["per_topic"].items():
        print(f"  {topic} (n={info['count']}): P@1={info['P@1']}")


if __name__ == "__main__":
    main()
