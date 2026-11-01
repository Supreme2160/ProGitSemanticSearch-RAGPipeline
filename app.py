import json
import os
import re

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()
try:
    api_key = st.secrets["GOOGLE_GEMINI_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_GEMINI_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# ---------- helpers (same logic as search.py) ----------

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def fuse(faiss_ids, bm25_ids, k=10):
    """Reciprocal rank fusion. Returns [(chunk_id, points), ...] best first."""
    points = {}
    for rank, cid in enumerate(faiss_ids, start=1):
        points[cid] = points.get(cid, 0) + 1 / (60 + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        points[cid] = points.get(cid, 0) + 1 / (60 + rank)
    ranked = sorted(points, key=points.get, reverse=True)[:k]
    return [(cid, points[cid]) for cid in ranked]


def format_pages(c):
    if c["startpage"] == c["endpage"]:
        return f"p. {c['startpage']}"
    return f"pp. {c['startpage']}-{c['endpage']}"


# ---------- load once, reuse on every click ----------

@st.cache_resource
def load_everything():
    vecs = np.load("vecs_800_200.npy")
    with open("chunks_800_200.json") as f:
        chunks = json.load(f)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return model, index, chunks, bm25


model, index, chunks, bm25 = load_everything()


# ---------- the three search methods ----------

def semantic(query, k=10):
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q, k)
    return list(zip(ids[0].tolist(), scores[0].tolist()))


def keyword(query, k=10):
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(scores)[-k:][::-1]
    return [(int(i), float(scores[i])) for i in top]


def hybrid(query, k=10):
    sem_ids = [cid for cid, _ in semantic(query, k)]
    kw_ids = [cid for cid, _ in keyword(query, k)]
    return fuse(sem_ids, kw_ids, k)


METHODS = {
    "Semantic (embeddings)": semantic,
    "Keyword (BM25)": keyword,
    "Hybrid (fusion)": hybrid,
}

# ---------- page ----------

st.set_page_config(page_title="Pro Git search", page_icon="🔎")
st.title("Pro Git: semantic search")
st.caption(
    "Ask a question in your own words. Results are passages from the book "
    "with printed page numbers."
)

method = st.radio("Search method", list(METHODS), horizontal=True)
query = st.text_input("Your question", placeholder="How do I undo my last commit?")

if query:
    results = METHODS[method](query)[:3]

    # 1. Gemini answer section (shown above the sources)
    if client is None:
        st.info("No Gemini API key found in .env, so answer generation is off.")
    elif st.button("Generate answer"):
        with st.spinner("Synthesizing answer from retrieved passages..."):
            context_blocks = "\n\n".join(
                f"[{format_pages(chunks[cid])}]:\n{chunks[cid]['text']}"
                for cid, _ in results
            )
            prompt = (
                "You are an assistant answering technical questions about Git using only the provided excerpts.\n"
                "Constraints:\n"
                "- Answer strictly using only the provided excerpts below.\n"
                "- Cite the relevant page numbers (e.g., p. 46 or pp. 229-230) inline when stating facts.\n"
                "- If the answer cannot be determined from the excerpts, respond with the exact phrase: 'Not found in the provided text.'\n\n"
                f"Context excerpts:\n{context_blocks}\n\n"
                f"Question: {query}\nAnswer:"
            )
            try:
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )
                st.subheader("Generated Answer")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    st.divider()

    # 2. Retrieved passages
    for rank, (cid, score) in enumerate(results, start=1):
        c = chunks[cid]
        pages = format_pages(c)
        label = f"#{rank}  |  {pages}  |  score {score:.3f}"
        with st.expander(label, expanded=(rank == 1)):
            st.text(c["text"])

    st.caption(
        "Hybrid scores are fusion points, not similarities, so they are not "
        "comparable to the other two methods."
    )

    st.caption(
    "Book text from *Pro Git* (2nd Ed.) by Scott Chacon and Ben Straub, "
    "licensed under CC BY-NC-SA 3.0 (https://git-scm.com/book)."
)