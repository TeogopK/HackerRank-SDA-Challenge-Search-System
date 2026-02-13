"""
Run this once to pre-compute all indices and embeddings.
The Streamlit app will load these files instantly instead of
training models on every startup.

Usage:
    python precompute.py
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import classla
import stopwordsiso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------------
# Configuration
# ----------------------------
BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ALPHA = 0.4
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHALLENGES_DIR = PROJECT_ROOT / "scraped_data" / "challenges"
LABELS_DIR = PROJECT_ROOT / "openai_annotations"
OUTPUT_DIR = PROJECT_ROOT / "precomputed"

SECTION_PATTERN = re.compile(
    r"(PROBLEM STATEMENT:|INPUT FORMAT:|OUTPUT FORMAT:|CONSTRAINTS:)"
)


# ----------------------------
# Text preprocessing (same as search_app.py)
# ----------------------------
def extract_sections(text: str) -> str:
    parts = SECTION_PATTERN.split(text)
    sections = {}
    for i, part in enumerate(parts):
        key = part.strip().rstrip(":")
        if key in ("PROBLEM STATEMENT", "INPUT FORMAT", "OUTPUT FORMAT"):
            if i + 1 < len(parts):
                sections[key] = parts[i + 1]
    kept = [sections[k] for k in ("PROBLEM STATEMENT", "INPUT FORMAT", "OUTPUT FORMAT") if k in sections]
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


def preprocess_tfidf(text: str, nlp_pipeline, stopwords: set) -> str:
    if not text:
        return ""
    text = extract_sections(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp_pipeline(text)
    lemmas = []
    for sent in doc.sentences:
        for w in sent.words:
            if w.lemma and w.lemma not in stopwords and w.lemma.isalpha():
                lemmas.append(w.lemma)
    return " ".join(lemmas)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Load models ----
    print("Loading CLASSLA pipeline...")
    nlp_pipeline = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)

    print("Loading BERT model...")
    bert_model = SentenceTransformer(BERT_MODEL)

    stopwords = set(stopwordsiso.stopwords("bg"))

    # ---- Build docs list ----
    print("Loading documents...")
    docs = []
    for txt_path in sorted(CHALLENGES_DIR.glob("*.txt")):
        label_path = LABELS_DIR / f"{txt_path.stem}.json"
        if not label_path.exists():
            continue

        raw_text = txt_path.read_text(encoding="utf-8")
        ann_meta = json.loads(label_path.read_text(encoding="utf-8"))
        ann = ann_meta.get("annotation", {})

        json_path = CHALLENGES_DIR / f"{txt_path.stem}.json"
        hr_meta = {}
        if json_path.exists():
            hr_meta = json.loads(json_path.read_text(encoding="utf-8"))

        docs.append({
            "slug": txt_path.stem,
            "raw_text": raw_text,
            "primary_topic": ann.get("primary_topic", "unknown"),
            "paradigm": ann.get("problem_paradigm", "unknown"),
            "name": hr_meta.get("name", txt_path.stem),
            "difficulty": hr_meta.get("difficulty_name", "Unknown"),
            "max_score": hr_meta.get("max_score", 0),
            "success_ratio": hr_meta.get("success_ratio", 0.0),
            "contest": hr_meta.get("contest_slug", "unknown"),
            "tags": hr_meta.get("tag_names", []),
            "preview": hr_meta.get("preview", ""),
            "total_submissions": hr_meta.get("total_count", 0),
            "solved_count": hr_meta.get("solved_count", 0),
        })

    print(f"Found {len(docs)} annotated challenges.")

    # ---- TF-IDF ----
    print("Building TF-IDF index (this takes a while due to lemmatization)...")
    tfidf_corpus = [preprocess_tfidf(d["raw_text"], nlp_pipeline, stopwords) for d in docs]
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(tfidf_corpus)

    # ---- BERT ----
    print("Building BERT embeddings...")
    bert_corpus = [preprocess_bert(d["raw_text"]) for d in docs]
    bert_embeddings = bert_model.encode(bert_corpus, show_progress_bar=True, batch_size=32)

    # ---- Similarity matrices ----
    print("Computing similarity matrices...")
    tfidf_sim_matrix = cosine_similarity(tfidf_matrix)
    bert_sim_matrix = cosine_similarity(bert_embeddings)
    hybrid_sim_matrix = (1 - DEFAULT_ALPHA) * tfidf_sim_matrix + DEFAULT_ALPHA * bert_sim_matrix

    # ---- Save everything ----
    print("Saving to precomputed/...")

    with open(OUTPUT_DIR / "docs.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    with open(OUTPUT_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    from scipy import sparse
    sparse.save_npz(OUTPUT_DIR / "tfidf_matrix.npz", tfidf_matrix)

    np.save(OUTPUT_DIR / "bert_embeddings.npy", bert_embeddings)
    np.save(OUTPUT_DIR / "hybrid_sim_matrix.npy", hybrid_sim_matrix)

    print("Done! Files saved:")
    for p in sorted(OUTPUT_DIR.glob("*")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:30s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
