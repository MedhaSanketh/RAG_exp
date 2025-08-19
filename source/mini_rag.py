
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # small,fast

# small database
docs = [
    {"id": "doc1", "text": "Sea turtles migrate long distances to lay eggs on beaches."},
    {"id": "doc2", "text": "Electric cars use large battery packs and recharge at charging stations."},
    {"id": "doc3", "text": "Python lists are mutable, while tuples are immutable."},
    {"id": "doc4", "text": "Rainforests are home to millions of insect species and help regulate climate."},
]

texts = [d["text"] for d in docs]
embeddings = model.encode(texts, convert_to_numpy=True)
# normalize for cosine
def normalize(m): return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
embeddings = normalize(embeddings)

def search(query, top_k=2):
    qv = model.encode([query], convert_to_numpy=True)
    qv = qv / (np.linalg.norm(qv) + 1e-10)
    sims = (embeddings @ qv.T).squeeze()   # cosine since vectors are normalized
    idx = np.argsort(sims)[::-1][:top_k]
    return [(docs[i]["id"], docs[i]["text"], float(sims[i])) for i in idx]

if __name__ == "__main__":
    while True:
        q = input("\nQuery (blank to exit): ").strip()
        if not q: break
        results = search(q, top_k=3)
        print("\nTop results:")
        for rid, text, score in results:
            print(f" - {rid} (score {score:.3f}): {text}")
