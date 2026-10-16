# eval_chunks.py
# PURPOSE: Compare chunk sizes. For each setting, build a temporary in-memory
#          database, run the test questions, and print one results table.
#          Uses no Gemini requests and does not touch chroma_db/.
import chromadb

from ingest import load_all_pdfs, make_chunks
from store import embed_model
from eval_retrieval import load_questions, QUESTIONS_FILE, MIN_SCORE

CONFIGS = [(200, 30), (400, 60), (600, 100), (1000, 150), (1500, 200)]  # (size, overlap)
TOP_K = 5


def build_collection(pages, size, overlap):
    chunks = make_chunks(pages, size, overlap)
    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection(
        name=f"eval_{size}", metadata={"hnsw:space": "cosine"})
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, batch_size=64).tolist()
    col.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"file": c["file"], "page": c["page"]} for c in chunks],
    )
    return col, len(chunks)


def evaluate(col, items):
    answerable = [i for i in items if not i["none"]]
    unanswerable = [i for i in items if i["none"]]
    hits = {1: 0, 3: 0, 5: 0}
    wrongly_rejected = 0
    best_real = []

    for it in answerable:
        q_emb = embed_model.encode(it["q"]).tolist()
        r = col.query(query_embeddings=[q_emb], n_results=TOP_K)
        metas = r["metadatas"][0]
        scores = [1 - d for d in r["distances"][0]]
        best_real.append(scores[0])
        if scores[0] < MIN_SCORE:
            wrongly_rejected += 1
        rank = next((pos for pos, m in enumerate(metas, start=1)
                     if m["file"] == it["file"] and m["page"] in it["pages"]), None)
        for k in hits:
            if rank is not None and rank <= k:
                hits[k] += 1

    rejected, best_none = 0, []
    for it in unanswerable:
        q_emb = embed_model.encode(it["q"]).tolist()
        r = col.query(query_embeddings=[q_emb], n_results=1)
        score = 1 - r["distances"][0][0]
        best_none.append(score)
        if score < MIN_SCORE:
            rejected += 1

    return {
        "hits": hits, "na": len(answerable),
        "wrongly_rejected": wrongly_rejected,
        "rejected": rejected, "nu": len(unanswerable),
        "lowest_real": min(best_real) if best_real else 0,
        "highest_none": max(best_none) if best_none else 0,
    }


def main():
    items = load_questions(QUESTIONS_FILE)
    pages = load_all_pdfs("data")
    rows = []
    for size, overlap in CONFIGS:
        print(f"Building chunk_size={size}, overlap={overlap} ...")
        col, n_chunks = build_collection(pages, size, overlap)
        rows.append((size, overlap, n_chunks, evaluate(col, items)))

    print("\n| Chunk size / overlap | Chunks | Top 1 | Top 3 | Top 5 | "
          "Real questions rejected | No-answer rejected | Lowest real score | Highest no-answer score |")
    print("|---|---|---|---|---|---|---|---|---|")
    for size, overlap, n, r in rows:
        na = r["na"]
        print(f"| {size} / {overlap} | {n} | {r['hits'][1]}/{na} | {r['hits'][3]}/{na} | "
              f"{r['hits'][5]}/{na} | {r['wrongly_rejected']}/{na} | "
              f"{r['rejected']}/{r['nu']} | {r['lowest_real']:.2f} | {r['highest_none']:.2f} |")


if __name__ == "__main__":
    main()