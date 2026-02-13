import re
import math
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List

import classla
import stopwordsiso

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =========================
# NLP pipeline
# =========================

nlp = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)

# =========================
# Config
# =========================

STOPWORDS = set(stopwordsiso.stopwords("bg"))
ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

TOP_K_RULES = 30
MIN_DOC_FREQ = 2
TEST_RATIO = 0.2
SEED = 42

CHALLENGES_DIR = str(PROJECT_ROOT / "scraped_data" / "challenges")
LABELS_DIR = str(PROJECT_ROOT / "openai_annotations")

LABEL_MAP = {
    "graph_traversal": "graph",
    "shortest_path": "graph",
    "minimum_spanning_tree": "graph",
    "eulerian_hamiltonian": "graph",
    "binary_search_tree": "tree_structure",
    "tree": "tree_structure",
    "linked_list": "linear_structure",
    "stack_queue": "linear_structure",
    "priority_queue": "linear_structure",
    "dynamic_programming": "dynamic_programming",
    "sorting": "sorting",
    "hash_table": "hash_table",
    "searching": "searching",
}

DROP_LABELS = {"string"}

# =========================
# Preprocessing
# =========================


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_features(text: str) -> Dict[str, List[str]]:
    doc = nlp(normalize(text))
    tokens, verbs = [], []

    for sent in doc.sentences:
        for w in sent.words:
            if not w.lemma or not w.lemma.isalpha():
                continue
            if w.lemma in STOPWORDS or w.upos not in ALLOWED_POS:
                continue
            tokens.append(w.lemma)
            if w.upos == "VERB":
                verbs.append(w.lemma)

    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    return {"tokens": tokens, "bigrams": bigrams, "verbs": verbs}


# =========================
# Dataset
# =========================


def load_labeled_dataset(challenges_dir, labels_dir):
    data = []

    for p in Path(challenges_dir).glob("*.txt"):
        jp = Path(labels_dir) / f"{p.stem}.json"
        if not jp.exists():
            continue

        text = p.read_text(encoding="utf-8")
        meta = json.loads(jp.read_text(encoding="utf-8"))
        ann = meta.get("annotation", {})
        raw = ann.get("primary_topic")

        if not raw or raw in DROP_LABELS:
            continue

        data.append(
            {
                "slug": p.stem,
                "text": text,
                "raw_label": raw,
                "label": LABEL_MAP.get(raw, raw),
                "key_phrases": ann.get("key_phrases", []),
            }
        )

    if not data:
        raise RuntimeError("No data loaded")

    return data


def train_test_split(dataset):
    rng = random.Random(SEED)
    by_label = defaultdict(list)

    for s in dataset:
        by_label[s["label"]].append(s)

    train, test = [], []
    for group in by_label.values():
        rng.shuffle(group)
        k = max(1, int(len(group) * TEST_RATIO))
        test.extend(group[:k])
        train.extend(group[k:])

    return train, test


# =========================
# Rule induction
# =========================


def build_label_statistics(dataset):
    stats = defaultdict(lambda: defaultdict(Counter))
    df = defaultdict(lambda: defaultdict(Counter))
    label_counts = Counter()

    for s in dataset:
        label = s["label"]
        feats = extract_features(s["text"])
        label_counts[label] += 1

        for t, vals in feats.items():
            stats[t][label].update(vals)
            for v in set(vals):
                df[t][label][v] += 1

        kp = [normalize(p) for p in s.get("key_phrases", [])]
        stats["key_phrases"][label].update(kp)
        for v in set(kp):
            df["key_phrases"][label][v] += 1

    return stats, label_counts, df


def cue_score(label, cue, cue_type, stats, label_counts):
    a = stats[cue_type][label][cue] + 1
    b = sum(stats[cue_type][label].values()) + 1
    p1 = a / b

    c = sum(stats[cue_type][l][cue] for l in label_counts) + 1
    d = sum(sum(stats[cue_type][l].values()) for l in label_counts) + 1
    p2 = c / d

    return math.log(p1) - math.log(p2)


def induce_rules(stats, label_counts, df):
    rules = defaultdict(dict)

    for cue_type in stats:
        for label in label_counts:
            scored = []
            for cue in stats[cue_type][label]:
                if df[cue_type][label][cue] < MIN_DOC_FREQ:
                    continue
                s = cue_score(label, cue, cue_type, stats, label_counts)
                if s > 0:
                    scored.append((cue, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            rules[label][cue_type] = scored[:TOP_K_RULES]

    return rules


# =========================
# Prediction
# =========================


def predict(text, rules, key_phrases=None):
    feats = extract_features(text)
    scores = defaultdict(float)
    norm = normalize(text)

    for label, rule in rules.items():
        for t, cues in rule.items():
            if t == "key_phrases":
                for c, w in cues:
                    if c in norm:
                        scores[label] += w
            else:
                fset = set(feats[t])
                for c, w in cues:
                    if c in fset:
                        scores[label] += w

    if key_phrases:
        kp = {normalize(p) for p in key_phrases}
        for label, rule in rules.items():
            for c, w in rule.get("key_phrases", []):
                if c in kp:
                    scores[label] += w

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# =========================
# Evaluation + Saving
# =========================


def evaluate_top_k(dataset, rules, k):
    ok = 0
    total = 0
    for s in dataset:
        scores = predict(s["text"], rules, s.get("key_phrases"))
        if scores and s["label"] in list(scores)[:k]:
            ok += 1
        total += 1
    return ok / total if total else 0.0


def cue_overlap(dataset, rules):
    vals = []
    for s in dataset:
        scores = predict(s["text"], rules, s.get("key_phrases"))
        if scores:
            vals.append(1.0 if s["label"] in scores else 0.0)
    return sum(vals) / len(vals) if vals else 0.0


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_predictions(dataset, rules, path, k=5):
    rows = []
    for s in dataset:
        scores = predict(s["text"], rules, s.get("key_phrases"))
        if scores:
            rows.append({"slug": s["slug"], "gold": s["label"], "top_k": list(scores.items())[:k]})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def save_rules(rules, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({l: {t: [(c, float(w)) for c, w in v] for t, v in r.items()} for l, r in rules.items()}, f, indent=2)


# =========================
# Main
# =========================


def main():
    dataset = load_labeled_dataset(CHALLENGES_DIR, LABELS_DIR)
    train, test = train_test_split(dataset)

    stats, label_counts, df = build_label_statistics(train)
    rules = induce_rules(stats, label_counts, df)

    metrics = {
        "top_1": evaluate_top_k(test, rules, 1),
        "top_3": evaluate_top_k(test, rules, 3),
        "top_5": evaluate_top_k(test, rules, 5),
        "cue_overlap": cue_overlap(test, rules),
    }

    print("\nEvaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    results_dir = PROJECT_ROOT / "results" / "cue_extraction"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, str(results_dir / "results_metrics.json"))
    save_predictions(test, rules, str(results_dir / "test_predictions.json"))
    save_rules(rules, str(results_dir / "induced_rules.json"))

    print(f"\nSaved to {results_dir}:")
    print("  results_metrics.json")
    print("  test_predictions.json")
    print("  induced_rules.json")


if __name__ == "__main__":
    main()
