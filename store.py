# store.py
# PURPOSE: Create or load the persistent ChromaDB vector database.
#          Embeds chunks only when the database is empty.
import os

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import extract_pdf, load_all_pdfs, make_chunks

DB_PATH = "chroma_db"
COLLECTION_NAME = "pdf_chunks"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OCR_ENABLED = os.environ.get("ENABLE_OCR", "0") == "1"

embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def get_collection(data_folder="data", use_ocr=OCR_ENABLED):
    """Return the ChromaDB collection, building it first if it is empty."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        print("Database is empty, reading PDFs and embedding chunks...")
        pages = load_all_pdfs(data_folder, use_ocr=use_ocr)
        chunks = make_chunks(pages)
        if not chunks:
            print(f"No readable PDFs found in '{data_folder}'. Starting with an empty database.")
            return collection
        texts = [c["text"] for c in chunks]
        embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"file": c["file"], "page": c["page"]} for c in chunks],
        )
        print("Stored chunks:", collection.count())
    else:
        print("Loaded existing database with", collection.count(),
              "chunks (no embedding needed)")
    return collection

def add_pdf(collection, pdf_path):
    """Add one PDF to the database. Returns a dict describing what happened.
    A file name that is already in the database is not added again."""
    name = os.path.basename(pdf_path)

    if collection.get(where={"file": name}, limit=1)["ids"]:
        return {"status": "exists", "file": name, "chunks": 0, "skipped_pages": 0}

    pages = extract_pdf(pdf_path, use_ocr=OCR_ENABLED)
    skipped = sum(1 for p in pages if p["source"] == "skipped_scanned")
    chunks = make_chunks(pages)

    if not chunks:   # nothing readable, usually a fully scanned PDF
        return {"status": "no_text", "file": name, "chunks": 0, "skipped_pages": skipped}

    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"file": c["file"], "page": c["page"]} for c in chunks],
    )
    return {"status": "added", "file": name, "chunks": len(chunks), "skipped_pages": skipped}


if __name__ == "__main__":
    collection = get_collection()
    # Quick search test (no Gemini needed)
    question = "What is hard voting?"
    q_emb = embed_model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=3)
    for text, meta, dist in zip(results["documents"][0],
                                results["metadatas"][0],
                                results["distances"][0]):
        print(f"score={1 - dist:.2f} | {meta['file']} | page {meta['page']}")
        print(text[:150])
        print("-" * 40)