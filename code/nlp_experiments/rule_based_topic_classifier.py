import re
import math
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

import classla
import stopwordsiso

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =========================
# NLP pipeline (Bulgarian)
# =========================

nlp = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)

# =========================
# Configuration
# =========================

STOPWORDS = set(stopwordsiso.stopwords("bg"))

ALLOWED_POS = {
    "NOUN",
    "PROPN",
    "ADJ",
    "VERB",
}

TOP_K_RULES = 30
MIN_DOC_FREQ = 2
TEST_RATIO = 0.2
SEED = 42

CHALLENGES_DIR = str(PROJECT_ROOT / "scraped_data" / "challenges")
LABELS_DIR = str(PROJECT_ROOT / "openai_annotations")

# Merge sparse/related classes into broader groups
# - Graph algorithms: graph_traversal + shortest_path + minimum_spanning_tree + eulerian_hamiltonian
# - Tree structures: binary_search_tree + tree
# - Linear structures: linked_list + stack_queue + priority_queue
# - Drop singletons: string (1 sample) — too few to learn from
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
# Labels to drop (too few samples to learn anything)
DROP_LABELS = {"string"}

# =========================
# Text preprocessing
# =========================


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_features(text: str) -> Dict[str, List[str]]:
    text = normalize(text)
    doc = nlp(text)

    tokens = []
    verbs = []

    for sent in doc.sentences:
        for word in sent.words:
            lemma = word.lemma
            upos = word.upos

            if not lemma:
                continue
            if not lemma.isalpha():
                continue
            if lemma in STOPWORDS:
                continue
            if upos not in ALLOWED_POS:
                continue

            tokens.append(lemma)

            if upos == "VERB":
                verbs.append(lemma)

    bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    return {
        "tokens": tokens,
        "bigrams": bigrams,
        "verbs": verbs,
    }


# =========================
# Dataset loading
# =========================


def load_labeled_dataset(challenges_dir: str, labels_dir: str) -> List[Dict]:
    dataset = []

    for txt_path in Path(challenges_dir).glob("*.txt"):
        slug = txt_path.stem
        label_path = Path(labels_dir) / f"{slug}.json"

        if not label_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")

        with open(label_path, encoding="utf-8") as f:
            label_json = json.load(f)

        annotation = label_json.get("annotation", {})
        raw_label = annotation.get("primary_topic")
        if not raw_label:
            continue

        # Drop labels with too few samples
        if raw_label in DROP_LABELS:
            continue

        # Map to merged label
        label = LABEL_MAP.get(raw_label, raw_label)

        # Extract key_phrases and key_verbs from OpenAI annotations
        key_phrases = annotation.get("key_phrases", [])
        key_verbs = annotation.get("key_verbs", [])

        dataset.append({
            "slug": slug,
            "label": label,
            "raw_label": raw_label,
            "text": text,
            "key_phrases": key_phrases,
            "key_verbs": key_verbs,
        })

    if not dataset:
        raise RuntimeError("No labeled data found")

    return dataset


def train_test_split(dataset: List[Dict], test_ratio: float = TEST_RATIO, seed: int = SEED) -> Tuple[List[Dict], List[Dict]]:
    """Stratified split — each label keeps its proportion in train and test."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in dataset:
        by_label[sample["label"]].append(sample)

    train, test = [], []
    for _, samples in by_label.items():
        rng.shuffle(samples)
        n_test = max(1, int(len(samples) * test_ratio))
        test.extend(samples[:n_test])
        train.extend(samples[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# =========================
# Rule learning
# =========================


def build_label_statistics(dataset):
    stats = {
        "tokens": defaultdict(Counter),
        "bigrams": defaultdict(Counter),
        "verbs": defaultdict(Counter),
        "key_phrases": defaultdict(Counter),
    }

    # Document frequency: how many docs in a label contain this cue
    doc_freq = {
        "tokens": defaultdict(Counter),
        "bigrams": defaultdict(Counter),
        "verbs": defaultdict(Counter),
        "key_phrases": defaultdict(Counter),
    }

    label_counts = Counter()

    for sample in dataset:
        label = sample["label"]
        feats = extract_features(sample["text"])
        label_counts[label] += 1

        for ftype in ["tokens", "bigrams", "verbs"]:
            stats[ftype][label].update(feats[ftype])
            # Track document frequency (unique cues per document)
            for cue in set(feats[ftype]):
                doc_freq[ftype][label][cue] += 1

        # Add key_phrases from OpenAI annotations
        kp = [normalize(p) for p in sample.get("key_phrases", [])]
        stats["key_phrases"][label].update(kp)
        for cue in set(kp):
            doc_freq["key_phrases"][label][cue] += 1

    return stats, label_counts, doc_freq


def cue_score(label, cue, cue_type, stats, label_counts):
    """PMI-style score: log P(cue|label) - log P(cue)"""
    # P(cue | label)
    cue_in_label = stats[cue_type][label][cue]
    total_in_label = sum(stats[cue_type][label].values())
    p_cue_given_label = (cue_in_label + 1) / (total_in_label + 1)

    # P(cue) across all labels
    cue_total = sum(stats[cue_type][l][cue] for l in label_counts)
    grand_total = sum(sum(stats[cue_type][l].values()) for l in label_counts)
    p_cue = (cue_total + 1) / (grand_total + 1)

    return math.log(p_cue_given_label) - math.log(p_cue)


def induce_rules(stats, label_counts, doc_freq, top_k=TOP_K_RULES, min_df=MIN_DOC_FREQ):
    rules = defaultdict(dict)

    for cue_type in stats:
        for label in label_counts:
            scored = []
            for cue in stats[cue_type][label]:
                # Skip cues that appear in fewer than min_df documents for this label
                if doc_freq[cue_type][label][cue] < min_df:
                    continue
                score = cue_score(label, cue, cue_type, stats, label_counts)
                scored.append((cue, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            # Only keep rules with positive PMI (cue is more likely in this label)
            rules[label][cue_type] = [(c, s) for c, s in scored[:top_k] if s > 0]

    return rules


# =========================
# Prediction
# =========================


def predict(text: str, rules, key_phrases=None):
    feats = extract_features(text)
    scores = defaultdict(float)

    for label, rule in rules.items():
        for cue_type, cues in rule.items():
            if cue_type == "key_phrases":
                # Match key_phrases against the normalized text
                norm_text = normalize(text)
                for cue, weight in cues:
                    if cue in norm_text:
                        scores[label] += weight
            else:
                feat_set = set(feats[cue_type])
                for cue, weight in cues:
                    if cue in feat_set:
                        scores[label] += weight

    # Also boost if the sample's own key_phrases match rule key_phrases
    if key_phrases:
        norm_kp = set(normalize(p) for p in key_phrases)
        for label, rule in rules.items():
            for cue, weight in rule.get("key_phrases", []):
                if cue in norm_kp:
                    scores[label] += weight

    if not scores:
        return {}

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# =========================
# Evaluation
# =========================


def evaluate(dataset, rules):
    correct = 0

    for sample in dataset:
        scores = predict(sample["text"], rules, key_phrases=sample.get("key_phrases"))
        if not scores:
            continue
        predicted = next(iter(scores))
        if predicted == sample["label"]:
            correct += 1

    return correct / len(dataset)


def collect_predictions(dataset, rules):
    results = []
    for sample in dataset:
        scores = predict(sample["text"], rules, key_phrases=sample.get("key_phrases"))
        predicted = next(iter(scores)) if scores else None
        results.append({
            "slug": sample["slug"],
            "true_label": sample["label"],
            "predicted_label": predicted,
            "correct": predicted == sample["label"],
            "scores": {k: round(v, 4) for k, v in scores.items()},
        })
    return results


def per_class_metrics(predictions):
    labels = sorted({p["true_label"] for p in predictions} | {p["predicted_label"] for p in predictions if p["predicted_label"]})
    metrics = {}
    for label in labels:
        tp = sum(1 for p in predictions if p["true_label"] == label and p["predicted_label"] == label)
        fp = sum(1 for p in predictions if p["true_label"] != label and p["predicted_label"] == label)
        fn = sum(1 for p in predictions if p["true_label"] == label and p["predicted_label"] != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return metrics


# =========================
# Save results
# =========================


def save_results(rules, predictions, class_metrics, accuracy, output_dir=str(PROJECT_ROOT / "results" / "pmi")):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    # Save learned rules
    rules_serializable = {}
    for label, rule in rules.items():
        rules_serializable[label] = {
            cue_type: [{"cue": cue, "weight": round(w, 4)} for cue, w in cues]
            for cue_type, cues in rule.items()
        }
    with open(out / "learned_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_serializable, f, ensure_ascii=False, indent=2)

    # Save per-sample predictions
    with open(out / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Save evaluation report
    report = {
        "accuracy": round(accuracy, 4),
        "num_samples": len(predictions),
        "num_correct": sum(1 for p in predictions if p["correct"]),
        "per_class": class_metrics,
    }
    with open(out / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {out.resolve()}/")


# =========================
# Debug
# =========================


def print_rules(rules):
    for label, rule in rules.items():
        print(f"\n{'='*40}")
        print(f"LABEL: {label}")
        for cue_type, cues in rule.items():
            if not cues:
                continue
            print(f"  {cue_type}:")
            for cue, score in cues:
                print(f"    {cue:30s} {score:+.3f}")


# =========================
# Main
# =========================


def main():
    dataset = load_labeled_dataset(CHALLENGES_DIR, LABELS_DIR)
    print(f"Loaded {len(dataset)} labeled samples")

    # Label distribution
    label_dist = Counter(s["label"] for s in dataset)
    print(f"Labels ({len(label_dist)}):")
    for label, count in label_dist.most_common():
        print(f"  {label:30s} {count}")

    # Train/test split
    train_set, test_set = train_test_split(dataset)
    print(f"\nTrain: {len(train_set)}, Test: {len(test_set)}")

    # Build model from training data only
    stats, label_counts, doc_freq = build_label_statistics(train_set)
    rules = induce_rules(stats, label_counts, doc_freq)

    print_rules(rules)

    # Evaluate on train
    train_preds = collect_predictions(train_set, rules)
    train_acc = sum(1 for p in train_preds if p["correct"]) / len(train_preds)
    print(f"\nTrain Accuracy: {round(train_acc, 3)}")

    # Evaluate on test
    test_preds = collect_predictions(test_set, rules)
    test_acc = sum(1 for p in test_preds if p["correct"]) / len(test_preds)
    print(f"Test Accuracy:  {round(test_acc, 3)}")

    class_metrics = per_class_metrics(test_preds)
    print("\nPer-class metrics (test set):")
    for label, m in sorted(class_metrics.items()):
        print(f"  {label:30s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  (n={m['support']})")

    # Macro averages
    supported = [m for m in class_metrics.values() if m["support"] > 0]
    macro_p = sum(m["precision"] for m in supported) / len(supported)
    macro_r = sum(m["recall"] for m in supported) / len(supported)
    macro_f1 = sum(m["f1"] for m in supported) / len(supported)
    print(f"\n  {'MACRO AVERAGE':30s}  P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}")

    save_results(rules, test_preds, class_metrics, test_acc)

    # Show a few example predictions
    print("\n--- Example predictions ---")
    for sample in test_set[:3]:
        scores = predict(sample["text"], rules, key_phrases=sample.get("key_phrases"))
        pred = next(iter(scores)) if scores else "N/A"
        marker = "OK" if pred == sample["label"] else "XX"
        print(f"  [{marker}] {sample['slug']:20s}  true={sample['label']:25s}  pred={pred}")


if __name__ == "__main__":
    main()
