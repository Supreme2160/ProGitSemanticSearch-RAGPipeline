import pymupdf
import bisect
from sentence_transformers import SentenceTransformer
import numpy as np
import json
doc = pymupdf.open("data/progit.pdf")
lst = []
for i in range(len(doc)):
    text = doc[i].get_text()
    lst.append(len(text))

lessthan10 = 0
between10and50 = 0
morethan50 = 0
thinpages =[]
for i,v in enumerate(lst):
    if (v<10):
        lessthan10+=1
        thinpages.append(i)
    if (10<v<50):
        between10and50+=1
        thinpages.append(i)
    if (v>50):
        morethan50+=1


#exclude these pages 
pages_excluded = [a for a in range(0,15)]
pages_excluded.extend([x for x in range(483,501)])
for item in thinpages:
    if (item not in pages_excluded):
        pages_excluded.append(item)

cleantext= []

for i in range(len(doc)):
    if (i not in pages_excluded):
        text = doc[i].get_text().strip()
        lst = text.splitlines()
        if (lst[-1].isdigit()):
            d1 = {}
            d1["printed"] = int(lst[-1])
            del lst[-1]
            new_text="\n".join(lst)
            d1["text"] = new_text
            d1["pdf_index"] = i
            cleantext.append(d1)


for item in cleantext:
    if item["pdf_index"]-item["printed"]!=5:
        print("wrong")

#print("No of pages by my arithmetic ",501-len(pages_excluded))
#print("No of items in cleantext",len(cleantext))
#print(cleantext[0])


#making the big book
book = ""
starts = []

for item in cleantext:
    pair = (len(book),item["printed"])
    starts.append(pair)
    book += item["text"]+"\n"

#print(starts[:3])
#print(len(book))
#print(len(cleantext[0]["text"]) + 1)

positions = [x[0] for x in starts]

#the chunking 

chunksize = 800 #size in characters
overlap = 200 #how much character overlap is needed  

def chunking(chunksize,overlap,book):
    chunks = []
    uid = 0
    for i in range(0,len(book),chunksize-overlap):
        item = {}
        item["id"] = uid;
        item["start"] = i
        item["end"] = min(i+chunksize,len(book))
        item["text"] = book[item["start"]:item["end"]]
        uid+=1
        startpage = starts[bisect.bisect_right(positions,item["start"])-1][1]
        endpage = starts[bisect.bisect_right(positions,item["end"]-1)-1][1]
        item["startpage"] = startpage
        item["endpage"] = endpage
        chunks.append(item)
    return chunks

chunks = chunking(chunksize,overlap,book)
# print(len(chunks))
# print(chunks[0])
# print(chunks[1000])
# print(chunks[522])

# print("last second chunk",len(chunks[-2]))
# print("\n")
# print("last fifth chunk",len(chunks[-5]))
# print("\n")
# print("last fourth chunk",len(chunks[-4]))
# print("\n")
# print("last chunk",len(chunks[-1]))


# print("Number of chunks",len(chunks))

# model = SentenceTransformer("all-MiniLM-L6-v2")

#example
# a, b, c = model.encode(["undo my last commit", "revert the previous change", "bake a chocolate cake"],normalize_embeddings=True)
# print(a @ b, a @ c)

# chunk_text = [x['text'] for x in chunks]
# vecs = model.encode(chunk_text,normalize_embeddings=True,show_progress_bar=True)
# print(vecs.shape)

# np.save("vecs_800_200.npy", vecs)
vecs = np.load("vecs_800_200.npy")

# with open("chunks_800_200.json", "w") as f:
#     json.dump(chunks, f)

with open("chunks_800_200.json") as f:
     chunks = json.load(f)

def pages_for(phrase):
    found = set()
    i = book.find(phrase)
    while i != -1:
        found.add(starts[bisect.bisect_right(positions, i) - 1][1])
        i = book.find(phrase, i + 1)
    return sorted(found)

# print(pages_for("git rm --cached"))
# print(pages_for("git stash"))
# print(pages_for("git fetch"))

import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from eval import eval_questions
vecs = np.load("vecs_800_200.npy")
with open("chunks_800_200.json") as f:
    chunks = json.load(f)
index = faiss.IndexFlatIP(384)
index.add(vecs)

model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [e["q"] for e in eval_questions]
q = model.encode(questions, normalize_embeddings=True)

score,ids = index.search(q,3)

print(score[0],ids[0])

# page_text = {item["printed"]: item["text"] for item in cleantext}
# import bisect

# flat = book.replace("\n", " ")   # same length as book, so positions still match
# page_text = {item["printed"]: item["text"] for item in cleantext}

# def page_of(pos):
#     return starts[bisect.bisect_right(positions, pos) - 1][1]

# def review(label, phrase, n=3):
#     print("=" * 70)
#     print(label, "| phrase:", phrase)
#     i = flat.find(phrase)
#     shown = 0
#     allpages = set()
#     while i != -1:
#         allpages.add(page_of(i))
#         if shown < n:
#             print(f"\n[page {page_of(i)}]", flat[max(0, i - 150): i + 450])
#             shown += 1
#         i = flat.find(phrase, i + 1)
#     print("\nall pages with hits:", sorted(allpages))

# to_check = [
#     ("unstage a file",        "git reset HEAD"),
#     ("fix commit message",    "commit --amend"),
#     ("find bug commit",       "git bisect"),
#     ("three states",          "three main states"),
#     ("detached HEAD",         "detached HEAD"),
#     ("fast-forward / 3-way",  "fast-forward"),
#     ("git grep",              "git grep"),
#     ("hooks",                 "hooks"),
#     ("squash",                "squash"),
# ]

# for label, phrase in to_check:
#     review(label, phrase)

# round2 = [
#     ("unstage",   "Unstaging a Staged File"),
#     ("three-way", "three-way merge"),
#     ("hooks",     "hooks subdirectory"),
#     ("squash",    "Squashing Commits"),
# ]
# for label, phrase in round2:
#     review(label, phrase, n=2)

# for p in [240, 241, 60, 263, 307, 426]:
#     print("=" * 70, "\nprinted page", p, "\n", page_text[p][:800])

