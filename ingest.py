# ingest.py
# PURPOSE: Read PDFs into pages (with optional OCR for scanned pages)
#          and split the pages into overlapping chunks.
import io
import os
import glob
import json

import pymupdf as fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ocr_reader = None


def _ocr_page(page):
    """Run OCR on one page image. EasyOCR is imported only when needed."""
    global _ocr_reader
    import numpy as np
    from PIL import Image
    import easyocr

    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    pix = page.get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    lines = _ocr_reader.readtext(np.array(img), detail=0, paragraph=True)
    return "\n".join(lines)


def extract_pdf(pdf_path, use_ocr=False):
    """Return a list of pages: {file, page, text, source}.
    Uses a saved <pdf>.json cache if it exists, so OCR never runs twice."""
    name = os.path.basename(pdf_path)
    cache = pdf_path + ".json"
    if os.path.exists(cache):
        print(f"Loaded from cache: {name}")
        with open(cache) as f:
            return json.load(f)

    doc = fitz.open(pdf_path)
    pages, skipped = [], 0
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        source = "text"
        if len(text) < 20:  # no real text, so this is a scanned page
            if use_ocr:
                text = _ocr_page(page)
                source = "ocr_page"
            else:
                text, source = "", "skipped_scanned"
                skipped += 1
        pages.append({"file": name, "page": page_number,
                      "text": text, "source": source})

    if skipped:
        print(f"WARNING: {name}: {skipped} scanned page(s) skipped (OCR is off).")
    else:
        with open(cache, "w") as f:
            json.dump(pages, f)
        print(f"Extracted and cached: {name}")
    return pages


def load_all_pdfs(folder="data", use_ocr=False):
    """Load every PDF in the folder into one list of pages."""
    all_pages = []
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        all_pages += extract_pdf(path, use_ocr=use_ocr)
    return all_pages


def make_chunks(pages, chunk_size=600, chunk_overlap=100):
    """Split pages into overlapping chunks. Each chunk keeps file + page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for p in pages:
        if not p["text"].strip():
            continue  # skip empty (skipped scanned) pages
        for i, piece in enumerate(splitter.split_text(p["text"])):
            chunks.append({
                "id": f'{p["file"]}_p{p["page"]}_c{i}',
                "file": p["file"],
                "page": p["page"],
                "text": piece,
            })
    return chunks


if __name__ == "__main__":
    pages = load_all_pdfs("data")
    chunks = make_chunks(pages)
    print("Pages:", len(pages))
    print("Chunks:", len(chunks))
    print("Example chunk:", chunks[0]["id"], "|", chunks[0]["text"][:100])