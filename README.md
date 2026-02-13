## HackerRank SDA Challenge Search System

### Prerequisites

- Python 3.10+
- Google Chrome / Chromium (for scraping only)
- A HackerRank account (for scraping only)

---

### Project Structure

```
IR/Project/
├── code/
│   ├── data_extractor/          # Scraping, annotation, simple TF-IDF
│   │   ├── scraper.py
│   │   ├── classifier.py
│   │   └── similarity_search.py
│   ├── nlp_experiments/         # NLP experiments
│   │   ├── cue_extraction.py
│   │   ├── intent_extraction.py
│   │   ├── rule_based_topic_classifier.py
│   │   └── similarity_search_nlp.py
│   ├── ir_experiments/          # IR experiments (BERT, hybrid)
│   │   ├── similarity_search_bert.py
│   │   └── similarity_search_hybrid.py
│   ├── streamlit_app/           # Web application
│   │   ├── search_app.py
│   │   └── precompute.py
│   └── utils/
│       ├── paths.py
│       └── setup_nlp.py
├── scraped_data/                # Scraped challenges and contests
├── openai_annotations/          # GPT-4o topic annotations
├── precomputed/                 # Pre-built indices for the app
├── results/                     # Evaluation outputs
├── media/                       # Documentation and presentations
├── requirements.txt
├── all_links.csv
└── .env
```

---

### Setup

```bash
py -m venv env
env\Scripts\activate
pip install -r requirements.txt
playwright install
py code/utils/setup_nlp.py
```

Create a `.env` file in the project root (only needed for scraping and classification):

```env
HACKERRANK_EMAIL=your_email
HACKERRANK_PASSWORD=your_password
OPENAI_API_KEY=your_openai_key
```

---

### Scraping

- Use the `all_links.csv` file to scrape contests via `py code/data_extractor/scraper.py`.
- If a new contest is added it will be scraped. Already scraped contests will be skipped.
- Annotate scraped challenges with topics/paradigms: `py code/data_extractor/classifier.py`

---

### Running the Application

Build the precomputed indices (TF-IDF matrix, BERT embeddings, similarity matrix):

```bash
py code/streamlit_app/precompute.py
```

Launch the Streamlit web interface:

```bash
streamlit run code/streamlit_app/search_app.py
```

The app provides four tabs: **Search** (free-text hybrid search), **Browse** (filterable table), **Topics** (statistics/visualizations), and **Practice** (recommendations based on solved problems).

---

### Running Evaluations

Each script performs leave-one-out evaluation and outputs metrics (P@1, P@3, P@5, MRR, MAP@5):

```bash
py code/nlp_experiments/similarity_search_nlp.py      # TF-IDF with Bulgarian lemmatization
py code/ir_experiments/similarity_search_bert.py       # BERT multilingual embeddings
py code/ir_experiments/similarity_search_hybrid.py     # Hybrid TF-IDF + BERT (alpha sweep)
```

Results are saved to `results/`.

Additional NLP experiments:

```bash
py code/nlp_experiments/cue_extraction.py              # Rule-based slot extraction
py code/nlp_experiments/rule_based_topic_classifier.py  # Learned rule induction (PMI)
py code/nlp_experiments/intent_extraction.py           # Intent slot extraction
```
