import re
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------
# Text preprocessing
# ----------------------------
def preprocess(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML if any
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


# ----------------------------
# Load scraped challenges (.txt)
# ----------------------------
def load_challenges(challenges_dir: str) -> List[Dict]:
    docs = []

    for path in Path(challenges_dir).glob("*.txt"):
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()

        docs.append(
            {
                "slug": path.stem,
                "name": path.stem,
                "text": preprocess(raw_text),
            }
        )

    if not docs:
        raise RuntimeError("No challenge TXT files found.")

    return docs


# ----------------------------
# Build TF-IDF index
# ----------------------------
def build_index(docs: List[Dict]):
    corpus = [d["text"] for d in docs]

    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),  # bigrams help a lot for problems
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


# ----------------------------
# Similarity search
# ----------------------------
def find_similar(
    query_text: str,
    docs: List[Dict],
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = 5,
):
    query_vec = vectorizer.transform([preprocess(query_text)])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "slug": docs[i]["slug"],
            "name": docs[i]["name"],
            "score": float(similarities[i]),
        }
        for i in top_indices
    ]


# ----------------------------
# Load query challenge (.txt)
# ----------------------------
def load_query_challenge(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return preprocess(f.read())


# ----------------------------
# Main demo
# ----------------------------
def main():
    challenges_dir = str(PROJECT_ROOT / "scraped_data" / "challenges")
    query_challenge_path = str(PROJECT_ROOT / "query_challenge.txt")

    print("Loading scraped challenges...")
    docs = load_challenges(challenges_dir)
    print(f"Loaded {len(docs)} challenges")

    print("Building TF-IDF index...")
    vectorizer, tfidf_matrix = build_index(docs)

    print("Loading query challenge...")
    query_text = load_query_challenge(query_challenge_path)

    print("\nMost similar challenges:\n")
    results = find_similar(
        query_text=query_text,
        docs=docs,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=5,
    )

    for rank, r in enumerate(results, 1):
        print(f"{rank}. {r['slug']}  (similarity: {r['score']:.3f})")


if __name__ == "__main__":
    main()
