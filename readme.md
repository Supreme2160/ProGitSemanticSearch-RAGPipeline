# Pro Git: Semantic, Keyword & Hybrid Search (RAG)

A retrieval and grounded question-answering system over *Pro Git (2nd Edition)*. This project compares dense semantic embeddings (FAISS), sparse lexical matching (BM25), and Reciprocal Rank Fusion (RRF) on a 15-query benchmark, with an optional grounded answer generation step powered by the Gemini API.

## Architecture & Pipeline

1. **Corpus Ingestion (`ingest.py`)**:
   - Extracts text from the 501-page *Pro Git* PDF, trimming front/back matter (viewer pages 1–15 and 484–501) and low-content pages (< 50 characters) to retain **468 pages**.
   - Maps exact printed page numbers using the verified offset `printed_page = python_index - 5`.
   - Segments the text into **1,348 chunks** using an 800-character sliding window with a 200-character overlap.
   - Encodes L2-normalized dense embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and saves `vecs_800_200.npy` and `chunks_800_200.json`.

2. **Retrieval & Evaluation (`search.py` / `eval.py`)**:
   - **Semantic Search**: Exact inner-product search (`faiss.IndexFlatIP`) over normalized vectors (cosine similarity).
   - **Keyword Search**: Okapi BM25 (`rank_bm25`) over regex-tokenized chunks.
   - **Hybrid Search**: Reciprocal Rank Fusion ($k = 60$) combining rankings from both methods.

3. **Interactive UI & Grounded Generation (`app.py`)**:
   - Streamlit web interface allowing side-by-side comparison of the top 3 retrieved chunks and their printed page spans.
   - Optional Gemini synthesis strictly grounded in the retrieved passages with inline page citations and a fallback for out-of-scope queries.

## Evaluation Results (15-Query Benchmark)

| Method | Hit@3 | Hit@10 | Notes |
| :--- | :---: | :---: | :--- |
| **Semantic (Embeddings)** | 8 / 15 (53.3%) | **12 / 15 (80.0%)** | Strongest on paraphrased and intent-based queries at Top 10 |
| **Keyword (BM25)** | **9 / 15 (60.0%)** | 10 / 15 (66.7%) | Strongest on exact command/flag queries (6/7 at Top 3) |
| **Hybrid (RRF)** | 8 / 15 (53.3%) | 11 / 15 (73.3%) | Balances lexical precision and semantic recall |

## Running Locally

1. Install dependencies:
   `pip install -r requirements.txt`
2. (Optional) To enable Gemini answer generation, create a `.env` file in the root directory containing:
   `GOOGLE_GEMINI_KEY=your_api_key_here`
3. Launch the Streamlit app:
   `streamlit run app.py`

## License & Attribution

The pre-chunked text in `chunks_800_200.json` is derived from ***Pro Git (2nd Edition)*** by Scott Chacon and Ben Straub, available at [https://git-scm.com/book](https://git-scm.com/book) and licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (CC BY-NC-SA 3.0)](https://creativecommons.org/licenses/by-nc-sa/3.0/).