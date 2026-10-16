# eval_retrieval.py
# PURPOSE: Measure search quality without using Gemini.
#   - For answerable questions: is the correct page in the top results?
#   - For NONE questions: does the similarity cutoff reject them?
from store import embed_model, get_collection

QUESTIONS_FILE = "eval_questions.txt"
MIN_SCORE = 0.35          # keep in sync with MIN_SCORE in generator.py
CUTOFFS = [1, 3, 5, 10]   # ask() retrieves 5 chunks; notes/exercises retrieve 10


def load_questions(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 2 and parts[1].upper() == "NONE":
                items.append({"q": parts[0], "none": True})
            elif len(parts) == 3:
                try:
                    pages = {int(p) for p in parts[2].split(",") if p.strip()}
                except ValueError:
                    print(f"Line {n}: bad page list, skipped: {line}")
                    continue
                items.append({"q": parts[0], "none": False,
                              "file": parts[1], "pages": pages})
            else:
                print(f"Line {n}: could not read, skipped: {line}")
    return items


def search(collection, question, k):
    q_emb = embed_model.encode(question).tolist()
    r = collection.query(query_embeddings=[q_emb], n_results=k)
    return r["metadatas"][0], [1 - d for d in r["distances"][0]]


def main():
    collection = get_collection()
    items = load_questions(QUESTIONS_FILE)
    answerable = [i for i in items if not i["none"]]
    unanswerable = [i for i in items if i["none"]]

    hits = {k: 0 for k in CUTOFFS}
    wrongly_rejected = 0

    print("\nANSWERABLE QUESTIONS (searching all documents)")
    for it in answerable:
        metas, scores = search(collection, it["q"], max(CUTOFFS))
        rank = next((pos for pos, m in enumerate(metas, start=1)
                     if m["file"] == it["file"] and m["page"] in it["pages"]), None)
        for k in CUTOFFS:
            if rank is not None and rank <= k:
                hits[k] += 1
        if scores[0] < MIN_SCORE:
            wrongly_rejected += 1
        status = f"hit at rank {rank}" if rank else "MISS"
        print(f"  [{status:>14}] best={scores[0]:.2f} | {it['q']}")
        if rank is None:
            top = ", ".join(f"{m['file']} p{m['page']}" for m in metas[:3])
            print(f"                  expected {it['file']} p{sorted(it['pages'])}; got {top}")

    print("\nQUESTIONS WITH NO ANSWER (should be rejected by the cutoff)")
    rejected = 0
    for it in unanswerable:
        _, scores = search(collection, it["q"], 1)
        ok = scores[0] < MIN_SCORE
        rejected += ok
        print(f"  [{'rejected' if ok else 'NOT REJECTED':>12}] best={scores[0]:.2f} | {it['q']}")

    na, nu = len(answerable), len(unanswerable)
    print("\nSUMMARY")
    print("| Metric | Result |")
    print("|---|---|")
    for k in CUTOFFS:
        print(f"| Hit rate, top {k} chunks | {hits[k]}/{na} ({hits[k] / max(na, 1):.0%}) |")
    print(f"| Real questions wrongly rejected by cutoff | {wrongly_rejected}/{na} |")
    print(f"| Unanswerable questions rejected | {rejected}/{nu} |")


if __name__ == "__main__":
    main()