import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import classla
import stopwordsiso

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =========================
# NLP pipeline
# =========================

nlp = classla.Pipeline("bg", processors="tokenize,pos,lemma", tokenize_no_ssplit=True)

STOPWORDS = set(stopwordsiso.stopwords("bg"))

# =========================
# Dataset paths
# =========================

CHALLENGES_DIR = str(PROJECT_ROOT / "scraped_data" / "challenges")
LABELS_DIR = str(PROJECT_ROOT / "openai_annotations")

# =========================
# Text normalization
# =========================


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def lemmatize(text: str):
    doc = nlp(text)
    lemmas = []
    for sent in doc.sentences:
        for w in sent.words:
            if not w.lemma:
                continue
            if w.lemma in STOPWORDS:
                continue
            if not w.lemma.isalpha():
                continue
            lemmas.append(w.lemma)
    return lemmas


# =========================
# Intent slot rules
# =========================

# NOTE: All cue words MUST be in lemma form (singular indefinite for nouns,
# masculine singular indefinite for adjectives) because extract_slots()
# matches against lemmatized text from classla.

SLOT_RULES = {
    # ── topic: matches classifier LABELS ──
    "topic": {
        "sorting": {
            "сортирам", "сортиране", "подреждам", "подредба",
            "нарастващ", "намаляващ", "пореден", "ранг",
            "медиана", "сливане", "разбърквам",
        },
        "searching": {
            "търся", "търсене", "намирам", "двоичен",
            "проверявам", "съществувам", "линеен",
        },
        "linked_list": {
            "списък", "свързан", "възел", "указател",
            "следващ", "предишен", "глава", "опашка",
        },
        "stack_queue": {
            "стек", "опашка", "дек", "последен",
            "първи", "добавям", "премахвам", "връщам",
        },
        "binary_search_tree": {
            "двоичен", "дърво", "търсене", "вмъквам",
            "изтривам", "ключ", "наредба", "балансиран",
            "ротация", "авл",
        },
        "tree": {
            "дърво", "възел", "корен", "лист", "поддърво",
            "родител", "наследник", "предшественик", "предък",
            "дълбочина", "височина", "ниво", "бинарен",
        },
        "hash_table": {
            "хеш", "таблица", "ключ", "колизия",
            "функция", "речник", "множество",
        },
        "dynamic_programming": {
            "оптимален", "максимален", "минимален",
            "подзадача", "мемоизация", "максимум", "минимум",
            "раница", "печалба", "стойност", "таблица",
            "преход", "състояние", "рекурсия",
        },
        "priority_queue": {
            "приоритетен", "приоритет", "купа", "пирамида",
            "извличам", "минимален", "максимален",
        },
        "graph_traversal": {
            "граф", "обход", "обхождам", "посещавам",
            "достигам", "преминавам", "обиколка",
            "връх", "ребро", "съсед", "съседен",
            "компонента", "свързаност", "цикъл",
            "двуделен", "подграф", "инцидентен", "смежен",
        },
        "shortest_path": {
            "кратък", "път", "разстояние", "достижим",
            "маршрут", "пътека", "тегло", "теглов",
            "релаксация", "граф", "връх", "ребро",
        },
        "minimum_spanning_tree": {
            "остов", "остовен", "покриващ", "свързвам",
            "минимален", "тегло", "граф", "ребро",
        },
        "eulerian_hamiltonian": {
            "ойлеров", "хамилтонов", "цикъл", "път",
            "обиколка", "обхождам", "всеки", "ребро", "връх",
        },
    },
    # ── paradigm: matches classifier PROBLEM_PARADIGMS ──
    "paradigm": {
        "construction": {
            "построявам", "конструирам", "създавам", "извеждам",
            "генерирам", "строя", "съставям", "отпечатвам",
            "изграждам", "образувам",
        },
        "decision_problem": {
            "възможен", "невъзможен", "съществувам", "проверявам",
            "може", "дали",
        },
        "optimization": {
            "оптимален", "максимален", "минимален",
            "максимум", "минимум", "най",
            "печалба", "цена", "стойност",
        },
        "counting": {
            "колко", "брой", "преброявам", "начин",
            "вариант", "комбинация", "пермутация",
            "факториел", "биномен", "модул",
        },
        "traversal": {
            "обход", "обхождам", "посещавам", "достигам",
            "преминавам", "обиколка",
        },
        "search_task": {
            "търся", "търсене", "намирам", "двоичен",
            "проверявам", "откривам",
        },
    },
    # ── direction: auxiliary slot ──
    "direction": {
        "undirected": {"неориентиран", "неорентиран", "ненасочен", "двупосочен"},
        "directed": {"ориентиран", "насочен", "еднопосочен"},
    },
}

BOOLEAN_RULES = {
    "weighted": {"тегло", "теглов", "цена", "тежест", "дължина"},
    "multiple_tests": {"тест", "заявка", "запитване", "набор"},
}

# =========================
# Slot extraction
# =========================


def extract_slots(text: str):
    text = normalize(text)
    lemmas = lemmatize(text)
    lemma_set = set(lemmas)

    slots = {}

    # Categorical slots
    for slot, values in SLOT_RULES.items():
        scores = {}
        for val, cues in values.items():
            scores[val] = sum(1 for c in cues if c in lemma_set)
        best = max(scores.items(), key=lambda x: x[1])
        slots[slot] = best[0] if best[1] > 0 else "unknown"

    # Boolean slots
    for slot, cues in BOOLEAN_RULES.items():
        slots[slot] = any(c in lemma_set for c in cues)

    return slots


# =========================
# Dataset loading
# =========================


def load_dataset():
    data = []

    for txt_path in Path(CHALLENGES_DIR).glob("*.txt"):
        label_path = Path(LABELS_DIR) / f"{txt_path.stem}.json"
        if not label_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        meta = json.loads(label_path.read_text(encoding="utf-8"))
        ann = meta.get("annotation", {})

        data.append({"slug": txt_path.stem, "text": text, "primary_topic": ann.get("primary_topic")})

    if not data:
        raise RuntimeError("No data loaded")

    return data


# =========================
# Weak evaluation
# =========================


def evaluate_topic(dataset, predictions):
    """Compare rule-based topic prediction against OpenAI primary_topic labels."""
    correct = 0
    total = 0

    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for s, p in zip(dataset, predictions):
        gold = s["primary_topic"] or "unknown"
        pred = p["topic"]
        total += 1

        if gold == pred:
            correct += 1
            per_label[gold]["tp"] += 1
        else:
            per_label[gold]["fn"] += 1
            per_label[pred]["fp"] += 1

    accuracy = correct / total if total else 0

    # Macro-averaged P/R/F1
    precisions, recalls, f1s = [], [], []
    for _label, counts in per_label.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    n = len(per_label) or 1
    return {
        "accuracy": round(accuracy, 3),
        "macro_precision": round(sum(precisions) / n, 3),
        "macro_recall": round(sum(recalls) / n, 3),
        "macro_f1": round(sum(f1s) / n, 3),
        "total": total,
    }


# =========================
# Main
# =========================


def main():
    dataset = load_dataset()

    predictions = []
    for sample in dataset:
        slots = extract_slots(sample["text"])
        predictions.append(slots)

    results_dir = PROJECT_ROOT / "results" / "intend_slots"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save slot predictions
    with open(results_dir / "intent_slots.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"slug": s["slug"], "slots": p} for s, p in zip(dataset, predictions)], f, ensure_ascii=False, indent=2
        )

    # Evaluate topic prediction against OpenAI labels
    topic_metrics = evaluate_topic(dataset, predictions)

    with open(results_dir / "slot_metrics.json", "w", encoding="utf-8") as f:
        json.dump(topic_metrics, f, indent=2)

    print("\nSlot extraction completed.")
    print(f"Saved to {results_dir}:")
    print("  intent_slots.json")
    print("  slot_metrics.json")
    print("\nTopic prediction metrics:")
    for k, v in topic_metrics.items():
        print(f"  {k}: {v}")

    # Show examples
    print("\nExamples:")
    for i in range(3):
        print(f"\n{dataset[i]['slug']}")
        print(predictions[i])


if __name__ == "__main__":
    main()
