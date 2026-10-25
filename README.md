# 🔍 Information Retrieval System

Coursework repository for the **Information Retrieval Systems** course
(*Sistem Temu Kembali Informasi*, STKI) at Universitas Dian Nuswantoro. It holds two kinds of work:

- **[`PRODUCT/`](PRODUCT/)**: the course's main project, *Bookshelf*, a book title search engine.
- **`WEEK-1` to `WEEK-7`**: weekly progress. Each session of the course has its own
  interactive lab that puts that week's lecture material into practice.

Every project is written in Python with Streamlit, runs on its own, and has a live demo and a
detailed README of its own.

## Table of contents

- [Repository structure](#repository-structure)
- [Product: Bookshelf](#product-bookshelf)
- [Weekly progress](#weekly-progress)
- [Learning path](#learning-path)
- [Tech stack](#tech-stack)
- [Running a project locally](#running-a-project-locally)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [References](#references)

---

## Repository structure

```
.
├── PRODUCT/                         # Main project: Bookshelf, book title search engine
├── WEEK-1-MINI-SEARCH-ENGINE/       # Session 1: Introduction to IR
├── WEEK-2-PREPROCESSING-LAB/        # Session 2: Document preprocessing
├── WEEK-3-BOOLEAN-EMBEDDINGS-LAB/   # Session 3: Boolean retrieval & word embeddings
├── WEEK-4-VECTOR-SPACE-LAB/         # Session 4: Vector space model & dense retrieval
├── WEEK-5-TERM-WEIGHTING-LAB/       # Session 5: Term weighting & learned sparse retrieval
├── WEEK-6-SEARCH-ENGINE-LAB/        # Session 6: Search engine architecture & hybrid search
└── WEEK-7-IR-EVALUATION-LAB/        # Session 7: Evaluating IR models
```

Each folder follows the same layout:

| Item | Contents |
|---|---|
| `app.py` (or `pencarian_buku.py`) | Streamlit app (user interface) |
| Other `.py` files | Retrieval logic, kept separate from the interface |
| `data/` | Collections, stopword lists, or pretrained vectors used by the app |
| `docs/` | Screenshots used in the README |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation: features, how it works, worked examples, limitations |

## Product: Bookshelf

📚 **[`PRODUCT/`](PRODUCT/)** · 🔗 **[Live demo](https://information-retrieval-system-kzqtwhfxy6mnmgcp8ns7el.streamlit.app/)**

A book title search engine built on the **Vector Space Model (TF-IDF + Cosine Similarity)**.
Type a letter or a word from a book title and the matching books appear on a bookshelf, ordered by
relevance. Click a book spine to see its cover and details.

- **Search as you type**, with no need to press Enter.
- **Partial-word search**: `asu` finds *Salah Asuhan*, and the matched part is highlighted on the spine.
- **Relevance ranking** by cosine similarity (leftmost book = rank 1).
- **Indonesian stemming** (`menceritakan` → `cerita`) and **English title support**
  (stopwords, plurals, accented letters).
- **Transparent process**: every retrieval step for the current query is shown, and the full
  inverted index can be browsed in the app.
- **Corpus of 82 books**: Indonesian novels, world classics, Japanese literature, and manga.

| File | Contents |
|---|---|
| `pencarian_buku.py` | Streamlit app (bookshelf interface) |
| `ir.py` | Retrieval engine: preprocessing, stemming, inverted index, TF-IDF, cosine similarity |
| `books.json` | Corpus: 82 books with title, author, genre, year, publisher, and cover |
| `requirements.txt` | Python dependencies |

Full documentation: [`PRODUCT/README.md`](PRODUCT/README.md)

## Weekly progress

One lab per session. Each one builds on the previous weeks, from a first mini search engine to a
full search engine and its evaluation.

| Week | Lab | Lecture topic | What it covers | Demo |
|:---:|---|---|---|:---:|
| 1 | [📇 Retrieval Catalog](WEEK-1-MINI-SEARCH-ENGINE/) | Introduction to Information Retrieval | Mini search engine for a book catalog with three models (Boolean, TF-IDF + cosine, BM25); shows query preprocessing, inverted index lookup, and scoring | [Open](https://information-retrieval-system-9mzr4yjzaakfymgiet859u.streamlit.app) |
| 2 | [🧪 Preprocessing Lab](WEEK-2-PREPROCESSING-LAB/) | Document Preprocessing | Cleaning, case folding, tokenization, stopword removal, phrase detection, and stemming step by step; rule-based stemming vs subword tokenization (BPE, WordPiece) | [Open](https://information-retrieval-system-lyjyd26zsei3epu5xzreuw.streamlit.app/) |
| 3 | [🧮 Boolean & Embeddings Lab](WEEK-3-BOOLEAN-EMBEDDINGS-LAB/) | Boolean Retrieval & Word Embeddings | `AND` / `OR` / `NOT` queries on an incidence matrix and an inverted index; GloVe similarity, nearest neighbors, and analogies; Boolean vs TF-IDF vs embeddings | [Open](https://information-retrieval-system-sde6wfb5iwdjyagex9edry.streamlit.app/) |
| 4 | [🧭 Vector Space Lab](WEEK-4-VECTOR-SPACE-LAB/) | Vector Space Model & Dense Retrieval | TF-IDF and cosine similarity number by number; cosine vs Euclidean; k-means clustering; bi-encoder (`all-MiniLM-L6-v2`) and hybrid retrieval | [Open](https://information-retrieval-system-qqvduqhixqpvypcnejfxwd.streamlit.app/) |
| 5 | [⚖️ Term Weighting Lab](WEEK-5-TERM-WEIGHTING-LAB/) | Term Weighting & Learned Sparse Retrieval | TF variants, BM25 term by term (effect of `k₁` and `b`), SPLADE, and BM25 + dense hybrid evaluated with P@5 and P@10 | [Open](https://information-retrieval-system-5mgtaeghbvah8u4n84vncb.streamlit.app/) |
| 6 | [🔎 Search Engine Lab](WEEK-6-SEARCH-ENGINE-LAB/) | Search Engine Architecture & Hybrid Search | Full pipeline on a simulated web of 37 pages: crawling, deduplication, indexing, query understanding, BM25 + dense fusion, reranking, ColBERT late interaction, and vector database indexes | [Open](https://information-retrieval-system-kthsppovkbbjgn2cbebfjc.streamlit.app/) |
| 7 | [📏 IR Evaluation Lab](WEEK-7-IR-EVALUATION-LAB/) | Evaluating IR Models: Classic & Modern Metrics | Precision, recall, F1, P@k, R@k, AP, MAP, MRR, nDCG on a TREC-style collection; significance tests; Cohen's κ; BEIR-style comparison and RAGAS-style RAG metrics | [Open](https://information-retrieval-system-6sot3vl4oedinqb6xb9xtv.streamlit.app/) |

## Learning path

```
Week 1  Mini search engine ──► the big picture: index, query, ranking
Week 2  Preprocessing       ──► how raw text becomes index terms
Week 3  Boolean & embeddings──► exact match vs meaning
Week 4  Vector space        ──► sparse, dense, and hybrid ranking
Week 5  Term weighting      ──► TF-IDF, BM25, SPLADE
Week 6  Search engine       ──► everything combined into one system
Week 7  Evaluation          ──► measuring how good a system is
                                    │
                                    ▼
PRODUCT  Bookshelf          ──► the concepts applied to a real search app
```

## Tech stack

| Tool | Used for |
|---|---|
| [Python](https://www.python.org/) 3.10+ | All projects |
| [Streamlit](https://streamlit.io/) | User interface and deployment |
| pandas, NumPy, Altair | Tables, computation, and charts |
| PySastrawi, NLTK, simplemma | Indonesian and English stemming, stopwords, lemmatization (Week 2, 4) |
| Hugging Face `tokenizers` | BPE and WordPiece subword tokenization (Week 2) |
| GloVe (50d) | Pretrained word embeddings (Week 3 to 7) |
| `sentence-transformers`, `transformers` | Dense retrieval and SPLADE (Week 4 to 7) |

## Running a project locally

Requires Python 3.10 or newer. Pick a folder, install its dependencies, and run its app:

```bash
git clone https://github.com/111202214411-ux/Information-Retrieval-System.git
cd Information-Retrieval-System/WEEK-1-MINI-SEARCH-ENGINE   # or any other folder
pip install -r requirements.txt
streamlit run app.py
```

For the product, use `PRODUCT/` and run `streamlit run pencarian_buku.py`.
The app opens in your browser at `http://localhost:8501`.

## Deploying to Streamlit Community Cloud

Every folder is deployed as a separate app from this same repository:

1. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
2. Select this repository and the `main` branch.
3. Set **Main file path** to the app inside the folder, for example
   `WEEK-1-MINI-SEARCH-ENGINE/app.py` or `PRODUCT/pencarian_buku.py`, then click **Deploy**.

Streamlit installs the `requirements.txt` from the same folder as the main file.
Apps update automatically whenever changes are pushed to GitHub.

## References

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*.
  Cambridge University Press.
- Croft, W. B., Metzler, D., & Strohman, T. (2010). *Search Engines: Information Retrieval in
  Practice*. Addison-Wesley.
- Lecture slides, Sessions 1 to 7, STKI Lecturer Team, Faculty of Computer Science,
  Universitas Dian Nuswantoro.

Each folder's README lists the references specific to that week.

---

Built by **Rizqi Aditya**, Informatics Engineering, Universitas Dian Nuswantoro,
for the Information Retrieval Systems course.
