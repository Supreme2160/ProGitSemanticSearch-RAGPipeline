import faiss
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
from eval import eval_questions
from rank_bm25 import BM25Okapi
vecs = np.load("vecs_800_200.npy")
with open("chunks_800_200.json") as f:
    chunks = json.load(f)
index = faiss.IndexFlatIP(384)
index.add(vecs)
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [e["q"] for e in eval_questions]
q = model.encode(questions, normalize_embeddings=True)
K =10
score,ids = index.search(q,K)
ranks = []
# print(score[0],ids[0])

# hits = 0
# for n,row in enumerate(ids):
#     labeled = eval_questions[n]["pages"]
#     hit = False     
#     for k in row:
#         for i in labeled:
#             if (chunks[k]["startpage"]<=i<=chunks[k]["endpage"]):
#                 hit = True
#     if hit:
#         hits+=1
#     else: 
#         print("MISS:",eval_questions[n]["q"])
# print(hits,"out of 15")



hits = 0
scored = 0
by_type = {}   # type -> [hits, total]

K = 10
score, ids = index.search(q, K)

def first_rank(id_list,labeled):
    rank = None
    for r,el in enumerate(id_list,start=1):
        if any(chunks[el]["startpage"]<=p<=chunks[el]["endpage"] for p in labeled):
            rank = r
            return rank
    return None


ranks = []   # rank of first correct chunk per question, or None
for n, row in enumerate(ids):
    labeled = eval_questions[n]["pages"]
    rank = None
    for r, k in enumerate(row, start=1):          # r = 1..10
        if any(chunks[k]["startpage"] <= p <= chunks[k]["endpage"] for p in labeled):
            rank = r
            break                                  # first hit is all we need
    ranks.append(rank)
    print(rank, "|", eval_questions[n]["type"], "|", eval_questions[n]["q"][:60])

hit3  = sum(1 for r in ranks if r is not None and r <= 3)
hit10 = sum(1 for r in ranks if r is not None)
print(f"hit@3: {hit3}/15   hit@10: {hit10}/15")

# //bm-25

def tokenize(text):
    return re.findall(r"\w+",text.lower())

bm25_ranks = []
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
bm25_ids = []
for n,question in enumerate(questions):
    scores = bm25.get_scores(tokenize(question))
    last10=np.argsort(scores)[-10:]
    last10 = (last10[::-1]).tolist()
    labeled = eval_questions[n]["pages"]
    rank = None
    bm25_ids.append(last10)
    for i,el in enumerate(last10,start=1):
        if any(chunks[el]["startpage"]<=p<=chunks[el]["endpage"] for p in labeled):
            rank = i
            break
    bm25_ranks.append(rank)

print(len(bm25_ranks))   # should be 15
bm25_hit3  = sum(1 for r in bm25_ranks if r is not None and r <= 3)
bm25_hit10 = sum(1 for r in bm25_ranks if r is not None)
print(f"BM25 hit@3: {bm25_hit3}/15   hit@10: {bm25_hit10}/15")

for e, f, b in zip(eval_questions, ranks, bm25_ranks):
    print(e["type"], "| FAISS:", f, "| BM25:", b, "|", e["q"][:50]) 

#hybrid approach
def fuse(faiss_row,bm25_row):
    chunktopoints ={}

    for i,el in enumerate(faiss_row,start=1):
        chunktopoints[el] = chunktopoints.get(el, 0) + 1 / (60 + i)

    for i,el in enumerate(bm25_row,start=1):
            chunktopoints[el] = chunktopoints.get(el, 0) + 1 / (60 + i)

    chunklist = sorted(chunktopoints,key=chunktopoints.get,reverse=True)
    return chunklist[0:10]

hybrid_ranks = []
for n in range(len(eval_questions)):
    fused = fuse(ids[n].tolist(), bm25_ids[n])
    labeled = eval_questions[n]["pages"]
    hybrid_ranks.append(first_rank(fused,labeled))

h3  = sum(1 for r in hybrid_ranks if r is not None and r <= 3)
h10 = sum(1 for r in hybrid_ranks if r is not None)
print(f"Hybrid hit@3: {h3}/15   hit@10: {h10}/15")