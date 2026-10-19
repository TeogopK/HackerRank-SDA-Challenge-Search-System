import os
import re
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ----------------------------
# Configuration
# ----------------------------
BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ALPHA = 0.4
PRECOMPUTED_DIR = Path(__file__).resolve().parent.parent.parent / "precomputed"

SECTION_PATTERN = re.compile(
    r"(PROBLEM STATEMENT:|INPUT FORMAT:|OUTPUT FORMAT:|CONSTRAINTS:)"
)


# ----------------------------
# Text preprocessing (for live search queries only)
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


# ----------------------------
# Fast loading from pre-computed files
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_index():
    with open(PRECOMPUTED_DIR / "docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    with open(PRECOMPUTED_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = sparse.load_npz(PRECOMPUTED_DIR / "tfidf_matrix.npz")
    bert_embeddings = np.load(PRECOMPUTED_DIR / "bert_embeddings.npy")
    hybrid_sim_matrix = np.load(PRECOMPUTED_DIR / "hybrid_sim_matrix.npy")
    slug_to_idx = {d["slug"]: i for i, d in enumerate(docs)}
    return docs, vectorizer, tfidf_matrix, bert_embeddings, hybrid_sim_matrix, slug_to_idx


# Lazy-loaded: only when the user actually searches
@st.cache_resource(show_spinner="Loading search models (first search only)...")
def load_search_models():
    # classla 1.1 checkpoints predate torch>=2.6's weights_only=True default for torch.load
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import classla
    import stopwordsiso
    from sentence_transformers import SentenceTransformer
    try:
        nlp_pipeline = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)
    except Exception:
        # Fresh environment (e.g. Streamlit Cloud): the Bulgarian model isn't downloaded yet
        classla.download("bg")
        nlp_pipeline = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)
    bert_model = SentenceTransformer(BERT_MODEL)
    stopwords = set(stopwordsiso.stopwords("bg"))
    return nlp_pipeline, bert_model, stopwords


def get_neighbors(idx: int, hybrid_sim_matrix, docs, top_k: int = 5):
    sims = hybrid_sim_matrix[idx].copy()
    sims[idx] = -1.0
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(docs[i], float(sims[i])) for i in top_indices]


def search_query(query: str, docs, vectorizer, tfidf_matrix, bert_embeddings, alpha: float):
    nlp_pipeline, bert_model, stopwords = load_search_models()

    query_tfidf = preprocess_tfidf(query, nlp_pipeline, stopwords)
    tfidf_vec = vectorizer.transform([query_tfidf])
    tfidf_sims = cosine_similarity(tfidf_vec, tfidf_matrix)[0]

    query_bert = preprocess_bert(query)
    bert_vec = bert_model.encode(query_bert)
    bert_sims = cosine_similarity([bert_vec], bert_embeddings)[0]

    scores = (1 - alpha) * tfidf_sims + alpha * bert_sims
    top_indices = np.argsort(scores)[::-1]

    results = []
    for rank, i in enumerate(top_indices, 1):
        doc = docs[i]
        results.append({
            "rank": rank, "slug": doc["slug"], "name": doc["name"],
            "score": float(scores[i]),
            "tfidf_score": float(tfidf_sims[i]),
            "bert_score": float(bert_sims[i]),
            "topic": doc["primary_topic"], "paradigm": doc["paradigm"],
            "difficulty": doc["difficulty"], "max_score": doc["max_score"],
            "success_ratio": doc["success_ratio"], "contest": doc["contest"],
            "tags": doc["tags"], "preview": doc["preview"],
            "total_submissions": doc["total_submissions"],
            "solved_count": doc["solved_count"], "raw_text": doc["raw_text"],
        })
    return results


def apply_filters(items, topic_filter, paradigm_filter, difficulty_filter, contest_filter):
    filtered = items
    if topic_filter:
        filtered = [r for r in filtered if r.get("topic", r.get("primary_topic")) in topic_filter]
    if paradigm_filter:
        filtered = [r for r in filtered if r.get("paradigm") in paradigm_filter]
    if difficulty_filter:
        filtered = [r for r in filtered if r.get("difficulty") in difficulty_filter]
    if contest_filter:
        filtered = [r for r in filtered if r.get("contest") in contest_filter]
    return filtered


# ----------------------------
# UI Components
# ----------------------------
def render_challenge_card(doc, score=None, show_similar_button=False, sim_matrix=None, docs=None, slug_to_idx=None):
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    with col1:
        slug = doc.get('slug', '')
        contest = doc.get('contest', '')
        name = doc.get('name', slug)
        hr_url = f"https://www.hackerrank.com/contests/{contest}/challenges/{slug}"
        st.markdown(f"**[{name}]({hr_url})** (`{slug}`)")
    with col2:
        topic = doc.get("topic", doc.get("primary_topic", ""))
        paradigm = doc.get("paradigm", "")
        st.markdown(f"`{topic}` · `{paradigm}`")
    with col3:
        if score is not None:
            st.markdown(f"**{score:.3f}**")
        else:
            st.markdown(f"{doc.get('success_ratio', 0) * 100:.0f}% solved")
    with col4:
        st.markdown(f"`{doc.get('difficulty', '')}`")

    tags_str = ", ".join(doc.get("tags", [])) or "—"
    success_pct = doc.get("success_ratio", 0) * 100
    st.caption(
        f"Contest: {doc.get('contest', '')} · "
        f"Score: {doc.get('max_score', '')} · "
        f"Success: {success_pct:.0f}% ({doc.get('solved_count', 0)}/{doc.get('total_submissions', 0)}) · "
        f"Tags: {tags_str}"
    )

    # Expandable: problem text + find similar
    with st.expander("Details"):
        if score is not None and "tfidf_score" in doc:
            c1, c2, c3 = st.columns(3)
            c1.metric("Hybrid", f"{score:.3f}")
            c2.metric("TF-IDF", f"{doc['tfidf_score']:.3f}")
            c3.metric("BERT", f"{doc['bert_score']:.3f}")
            st.divider()

        st.text(doc.get("raw_text", ""))

        if show_similar_button and sim_matrix is not None and docs is not None and slug_to_idx is not None:
            slug = doc.get("slug", "")
            if slug in slug_to_idx:
                st.divider()
                st.markdown("**Similar problems:**")
                idx = slug_to_idx[slug]
                neighbors = get_neighbors(idx, sim_matrix, docs, top_k=5)
                for nb_doc, nb_score in neighbors:
                    st.markdown(
                        f"- **{nb_doc['name']}** (`{nb_doc['slug']}`) — "
                        f"`{nb_doc['primary_topic']}` · `{nb_doc['difficulty']}` · "
                        f"similarity: {nb_score:.3f}"
                    )


def inline_filters(docs, prefix=""):
    with st.expander("Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            all_topics = sorted(set(d["primary_topic"] for d in docs))
            topic_filter = st.multiselect("Topic", options=all_topics, key=f"{prefix}_topic_f")
        with col2:
            all_paradigms = sorted(set(d["paradigm"] for d in docs))
            paradigm_filter = st.multiselect("Paradigm", options=all_paradigms, key=f"{prefix}_paradigm_f")
        with col3:
            all_difficulties = sorted(set(d["difficulty"] for d in docs))
            difficulty_filter = st.multiselect("Difficulty", options=all_difficulties, key=f"{prefix}_diff_f")
        with col4:
            all_contests = sorted(set(d["contest"] for d in docs))
            contest_filter = st.multiselect("Contest", options=all_contests, key=f"{prefix}_contest_f")
    return topic_filter, paradigm_filter, difficulty_filter, contest_filter


# ----------------------------
# Tab: Search
# ----------------------------
def tab_search(docs, vectorizer, tfidf_matrix, bert_embeddings, sim_matrix, slug_to_idx):
    st.header("Search")

    alpha = st.slider(
        "BERT weight (alpha)", 0.0, 1.0, DEFAULT_ALPHA, 0.1,
        help="0.0 = pure TF-IDF, 1.0 = pure BERT",
    )
    top_k = st.slider("Results", 1, 20, 5, key="search_k")
    st.caption(f"Blend: {(1 - alpha) * 100:.0f}% TF-IDF + {alpha * 100:.0f}% BERT")

    query = st.text_area(
        "Enter your search query", height=120,
        placeholder="Paste a problem statement or type keywords like 'сортиране на масив'...",
    )

    topic_f, paradigm_f, diff_f, contest_f = inline_filters(docs, prefix="search")

    if st.button("Search", type="primary") and query.strip():
        with st.spinner("Searching..."):
            results = search_query(query, docs, vectorizer, tfidf_matrix, bert_embeddings, alpha)

        filtered = apply_filters(results, topic_f, paradigm_f, diff_f, contest_f)
        shown = filtered[:top_k]

        st.subheader(f"Top {len(shown)} results")
        for r in shown:
            render_challenge_card(r, score=r["score"], show_similar_button=True,
                                  sim_matrix=sim_matrix, docs=docs, slug_to_idx=slug_to_idx)
            st.divider()

        if not shown:
            st.warning("No results match the current filters.")


# ----------------------------
# Tab: Browse
# ----------------------------
def tab_browse(docs, sim_matrix, slug_to_idx):
    st.header("Browse Challenges")

    topic_f, paradigm_f, diff_f, contest_f = inline_filters(docs, prefix="browse")
    filtered = apply_filters(docs, topic_f, paradigm_f, diff_f, contest_f)

    # Build dataframe for sortable table
    rows = []
    for d in filtered:
        rows.append({
            "Name": d["name"],
            "Slug": d["slug"],
            "Topic": d["primary_topic"],
            "Paradigm": d["paradigm"],
            "Difficulty": d["difficulty"],
            "Success %": round(d["success_ratio"] * 100, 1),
            "Max Score": d["max_score"],
            "Contest": d["contest"],
            "Submissions": d["total_submissions"],
        })

    df = pd.DataFrame(rows)

    st.caption(f"Showing {len(df)} challenges")

    sort_col = st.selectbox("Sort by", options=df.columns.tolist(), index=0)
    sort_asc = st.checkbox("Ascending", value=True)
    df_sorted = df.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    # Clickable table
    st.dataframe(df_sorted, use_container_width=True, height=400)

    # Detail view
    st.divider()
    slug_list = [d["slug"] for d in filtered]
    name_list = [f"{d['name']} ({d['slug']})" for d in filtered]

    if name_list:
        selected = st.selectbox("Select a challenge to view details", options=range(len(name_list)),
                                format_func=lambda i: name_list[i])
        doc = filtered[selected]
        render_challenge_card(doc, show_similar_button=True,
                              sim_matrix=sim_matrix, docs=docs, slug_to_idx=slug_to_idx)


# ----------------------------
# Tab: Topics
# ----------------------------
def tab_topics(docs, sim_matrix, slug_to_idx):
    st.header("Topic Dashboard")

    # Topic stats
    topic_counts = Counter(d["primary_topic"] for d in docs)
    topics_sorted = sorted(topic_counts.items(), key=lambda x: -x[1])

    # Bar chart
    chart_df = pd.DataFrame(topics_sorted, columns=["Topic", "Count"])
    st.bar_chart(chart_df.set_index("Topic"))

    # Topic selector
    selected_topic = st.selectbox("Select a topic", options=[t for t, _ in topics_sorted],
                                  format_func=lambda t: f"{t} ({topic_counts[t]} problems)")

    topic_docs = [d for d in docs if d["primary_topic"] == selected_topic]

    # Stats for selected topic
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Problems", len(topic_docs))
    difficulties = Counter(d["difficulty"] for d in topic_docs)
    c2.metric("Easy / Medium / Hard", f"{difficulties.get('Easy', 0)} / {difficulties.get('Medium', 0)} / {difficulties.get('Hard', 0)}")
    avg_success = np.mean([d["success_ratio"] for d in topic_docs]) * 100
    c3.metric("Avg Success Rate", f"{avg_success:.0f}%")
    contests = set(d["contest"] for d in topic_docs)
    c4.metric("Contests", len(contests))

    # Difficulty distribution
    st.subheader(f"Difficulty distribution for {selected_topic}")
    diff_df = pd.DataFrame([
        {"Difficulty": diff, "Count": count}
        for diff, count in sorted(difficulties.items())
    ])
    if not diff_df.empty:
        st.bar_chart(diff_df.set_index("Difficulty"))

    # Success rate distribution
    st.subheader("Success rate distribution")
    success_rates = [d["success_ratio"] * 100 for d in topic_docs]
    counts, bin_edges = np.histogram(success_rates, bins=10, range=(0, 100))
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}%" for i in range(len(counts))]
    hist_df = pd.DataFrame({"Range": bin_labels, "Count": counts}).set_index("Range")
    st.bar_chart(hist_df)

    # Problems list sorted by difficulty (easy first)
    st.subheader(f"All {selected_topic} problems (easiest first)")
    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2, "Unknown": 3}
    topic_docs_sorted = sorted(topic_docs, key=lambda d: (difficulty_order.get(d["difficulty"], 3), -d["success_ratio"]))

    for d in topic_docs_sorted:
        render_challenge_card(d, show_similar_button=True,
                              sim_matrix=sim_matrix, docs=docs, slug_to_idx=slug_to_idx)
        st.divider()


# ----------------------------
# Tab: Practice
# ----------------------------
def tab_practice(docs, sim_matrix, slug_to_idx):
    st.header("Practice Recommender")
    st.caption("Select problems you've already solved, and get recommendations for what to try next.")

    # Multi-select solved problems
    all_names = [f"{d['name']} ({d['slug']})" for d in docs]
    solved_indices = st.multiselect(
        "Problems you've solved",
        options=range(len(docs)),
        format_func=lambda i: all_names[i],
    )

    if not solved_indices:
        st.info("Select at least one solved problem to get recommendations.")
        return

    solved_slugs = {docs[i]["slug"] for i in solved_indices}
    solved_topics = Counter(docs[i]["primary_topic"] for i in solved_indices)

    # Show coverage
    all_topics = Counter(d["primary_topic"] for d in docs)
    st.subheader("Your topic coverage")

    coverage_rows = []
    for topic, total in sorted(all_topics.items(), key=lambda x: -x[1]):
        done = solved_topics.get(topic, 0)
        coverage_rows.append({
            "Topic": topic,
            "Solved": done,
            "Total": total,
            "Coverage": f"{done / total * 100:.0f}%",
        })
    coverage_df = pd.DataFrame(coverage_rows)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    # Find weakest topics
    weak_topics = sorted(coverage_rows, key=lambda r: r["Solved"] / r["Total"])

    st.subheader("Recommended next problems")
    st.caption("For each of your solved problems, the closest unsolved challenge in the same topic.")

    recommendations = []
    seen = set()

    for solved_idx in solved_indices:
        solved_doc = docs[solved_idx]
        topic = solved_doc["primary_topic"]
        sims = sim_matrix[solved_idx].copy()
        sims[solved_idx] = -1.0

        # Rank by similarity, pick first unsolved in same topic
        ranked = np.argsort(sims)[::-1]
        for candidate_idx in ranked:
            candidate = docs[candidate_idx]
            if candidate["slug"] in solved_slugs:
                continue
            if candidate["slug"] in seen:
                continue
            if candidate["primary_topic"] != topic:
                continue
            recommendations.append({
                "based_on": solved_doc["name"],
                "based_on_slug": solved_doc["slug"],
                "doc": candidate,
                "similarity": float(sims[candidate_idx]),
            })
            seen.add(candidate["slug"])
            break

    # Also recommend from weak topics
    st.divider()
    st.subheader("Strengthen weak topics")
    for row in weak_topics[:3]:
        topic = row["Topic"]
        if row["Solved"] == row["Total"]:
            continue

        # Find easiest unsolved in this topic
        topic_docs = [
            (i, d) for i, d in enumerate(docs)
            if d["primary_topic"] == topic and d["slug"] not in solved_slugs
        ]
        if not topic_docs:
            continue

        difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2, "Unknown": 3}
        topic_docs.sort(key=lambda x: (difficulty_order.get(x[1]["difficulty"], 3), -x[1]["success_ratio"]))
        easiest_idx, easiest_doc = topic_docs[0]

        st.markdown(f"**{topic}** — {row['Solved']}/{row['Total']} solved → try:")
        render_challenge_card(easiest_doc, show_similar_button=True,
                              sim_matrix=sim_matrix, docs=docs, slug_to_idx=slug_to_idx)
        st.divider()

    if recommendations:
        st.subheader("Similar to what you've solved")
        for rec in recommendations:
            st.markdown(f"Because you solved **{rec['based_on']}** (similarity: {rec['similarity']:.3f}):")
            render_challenge_card(rec["doc"], score=rec["similarity"], show_similar_button=True,
                                  sim_matrix=sim_matrix, docs=docs, slug_to_idx=slug_to_idx)
            st.divider()


# ----------------------------
# Main
# ----------------------------
def main():
    st.set_page_config(page_title="IR Challenge Search", layout="wide", initial_sidebar_state="collapsed")

    # Hide sidebar completely
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)

    docs, vectorizer, tfidf_matrix, bert_embeddings, sim_matrix, slug_to_idx = load_index()

    st.title("Algorithm Challenge Search")
    st.caption(f"Hybrid TF-IDF + BERT similarity search over {len(docs)} HackerRank algorithm problems")

    tab1, tab2, tab3, tab4 = st.tabs(["Search", "Browse", "Topics", "Practice"])

    with tab1:
        tab_search(docs, vectorizer, tfidf_matrix, bert_embeddings, sim_matrix, slug_to_idx)
    with tab2:
        tab_browse(docs, sim_matrix, slug_to_idx)
    with tab3:
        tab_topics(docs, sim_matrix, slug_to_idx)
    with tab4:
        tab_practice(docs, sim_matrix, slug_to_idx)


if __name__ == "__main__":
    main()
