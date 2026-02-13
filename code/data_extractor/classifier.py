import json
import os
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv

# ============================
# CONFIG
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"  # cheapest reliable option
TEMPERATURE = 0

LABELS = [
    "sorting",
    "searching",
    "linked_list",
    "stack_queue",
    "binary_search_tree",
    "tree",
    "hash_table",
    "dynamic_programming",
    "priority_queue",
    "graph_traversal",
    "shortest_path",
    "minimum_spanning_tree",
    "eulerian_hamiltonian",
]

MAX_SECONDARY = 3
MAX_KEY_PHRASES = 5


# ============================
# CLIENT
# ============================

client = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an NLP annotation assistant.
You classify programming problem descriptions into algorithmic topics.
You must follow the output schema exactly.
Return valid JSON only.
Do not explain anything.
"""

PROBLEM_PARADIGMS = [
    "construction",
    "decision_problem",
    "optimization",
    "counting",
    "traversal",
    "search_task",
]


def build_prompt(task_text: str) -> str:
    return f"""
You are classifying programming problems for a university algorithms course.

ALLOWED primary and secondary topics (choose ONLY from this list):
{LABELS}

IMPORTANT RULES:
- You MUST choose exactly ONE primary_topic from the list above
- You MUST NOT invent new topics
- If the problem is about trees, use "binary_search_tree"
- secondary_topics must also come ONLY from the list

Problem paradigms (choose exactly ONE):
{PROBLEM_PARADIGMS}

Paradigm rules:
- paradigm describes WHAT the task asks to do, not difficulty
- examples:
  traversal → visit nodes
  decision_problem → yes/no or true/false
  construction → build or output a structure
  optimization → min/max
  counting → count number of solutions

Key phrases:
- up to 5 short noun phrases
- must appear in the text

Key verbs:
- up to 5 verbs
- must appear in the text
- imperative or infinitive form

OUTPUT VALID JSON ONLY:

{{
  "primary_topic": "...",
  "secondary_topics": ["..."],
  "problem_paradigm": "...",
  "key_phrases": ["..."],
  "key_verbs": ["..."]
}}

Problem description:
\"\"\"
{task_text}
\"\"\"
"""


# ============================
# CLASSIFICATION
# ============================


def classify_task(task_text: str) -> Dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task_text)},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON returned:\n{content}") from e


# ============================
# DATASET LABELING (PER TASK)
# ============================


def label_dataset(
    input_dir: str,
    output_dir: str,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in input_dir.glob("*.txt"):
        try:
            print(f"Labeling {path.name} ...")

            text = path.read_text(encoding="utf-8")
            annotation = classify_task(text)

            output = {
                "id": path.stem,
                "annotation": annotation,
            }

            out_path = output_dir / f"{path.stem}.json"
            out_path.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Error processing {path.name}: {e}")


# ============================
# MAIN
# ============================


def main():
    label_dataset(
        input_dir=str(PROJECT_ROOT / "scraped_data" / "challenges"),
        output_dir=str(PROJECT_ROOT / "openai_annotations"),
    )


if __name__ == "__main__":
    main()
