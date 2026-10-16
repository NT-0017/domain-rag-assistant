# Domain RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions from your own PDF documents. It finds the relevant passages by meaning, then uses Google Gemini to write a new answer in its own words, with the source file and pages shown for every answer.

> Status: working prototype, developed and tested in GitHub Codespaces with Python 3.14.2. Deployment and OCR are still to do (see Roadmap).

## What it can do

- **Ask questions** about your PDFs and get an explained answer with sources
- **Page requests:** "Summarize page 4", "Explain page 4 as if I am 5"
- **Study material:** "Generate notes for ...", "Generate a True False exercise on ..."
- **Follow-up questions:** "Explain that more simply" is rewritten into a standalone question using the recent chat before searching
- **Says when it does not know:** if the best match is too weak, it replies "I could not find this in the documents." without calling Gemini
- **Multiple PDFs**, uploaded from the browser, with a "Search in" filter to ask about one document only
- **Clear errors:** a busy Gemini is retried automatically (2 retries); a used-up quota is reported plainly

## How it works

```
PDF -> extract text -> split into chunks (600 characters, 100 overlap)
    -> embed chunks -> store in ChromaDB (saved on disk)

Question -> (rewrite if it is a follow-up) -> embed -> find the most similar chunks
         -> best score below 0.35?  yes: "I could not find this in the documents."
                                    no:  Gemini writes the answer from the chunks
         -> answer + sources
```

| Request type | How it is handled |
|---|---|
| Normal question | 5 most similar chunks go to Gemini |
| "page N" request | The chunks of that page are fetched directly, with no search |
| Notes / exercises | 10 most similar chunks and a special prompt |

| Step | Tool |
|---|---|
| PDF reading | PyMuPDF |
| Chunking | LangChain text splitter |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector database | ChromaDB (persistent, cosine similarity) |
| Answer generation | Google Gemini (`gemini-3.5-flash-lite`) |
| Interface | Streamlit |

## Project structure

```
domain-rag-assistant/
├── app.py             # Streamlit chat interface
├── generator.py       # Gemini calls, retry, ask / smart_ask / assistant
├── store.py           # embeddings, ChromaDB, add_pdf
├── ingest.py          # PDF loading, optional OCR, chunking
├── eval_retrieval.py  # retrieval test (no LLM calls)
├── eval_chunks.py     # chunk size comparison (no LLM calls)
├── eval_answers.py    # runs a few questions through the full pipeline
├── check_scores.py    # prints similarity scores to choose the cutoff
├── eval_questions.txt # test questions: question | file | page
├── requirements.txt
├── .streamlit/config.toml   # theme
├── data/              # your PDFs (not included in the repo)
└── chroma_db/         # created automatically (not included in the repo)
```

## Setup

```bash
git clone https://github.com/NT-0017/domain-rag-assistant.git
cd domain-rag-assistant
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from [Google AI Studio](https://aistudio.google.com) and set it as an environment variable. Never commit the key.

```bash
export GEMINI_API_KEY="your-key-here"       # PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

In GitHub Codespaces, add it as a Codespaces secret named `GEMINI_API_KEY` instead.

Create a `data/` folder and put at least one text-based PDF in it, then run:

```bash
streamlit run app.py
```

The first run reads the PDFs in `data/` and builds the database in `chroma_db/`. Later runs load it instantly. After that you can add more PDFs with the upload box in the sidebar.

To test the pieces without the interface:

```bash
python ingest.py      # page and chunk counts
python store.py       # builds the database and runs a sample search
python generator.py   # asks one question through Gemini
```

## Evaluation

Three PDFs were used: a book chapter on ensemble learning (11 pages, scanned and read with OCR), a short machine learning guide (10 pages) and an interview preparation document (53 pages). They are not included in this repository, so the exact numbers cannot be reproduced without the same files, but the scripts and the method can be reused with your own documents and your own `eval_questions.txt`.

### 1. Retrieval (`python eval_retrieval.py`, no LLM calls)

15 answerable questions with known pages and 4 questions the documents cannot answer. A hit means the correct page is among the retrieved chunks.

| Metric | Result |
|---|---|
| Hit rate, top 1 | 10/15 (67%) |
| Hit rate, top 3 | 13/15 (87%) |
| Hit rate, top 5 | 15/15 (100%) |
| Real questions wrongly rejected by the cutoff (0.35) | 0/15 |
| Unanswerable questions rejected | 4/4 |

The app sends 5 chunks to Gemini, so the top-5 row is the one that matters most.

### 2. Chunk size comparison (`python eval_chunks.py`, no LLM calls)

Same questions, a separate temporary database for each setting.

| Chunk size / overlap | Chunks | Top 1 | Top 3 | Top 5 | Lowest real score | Highest no-answer score |
|---|---|---|---|---|---|---|
| 200 / 30 | 633 | 11/15 | 12/15 | 13/15 | 0.61 | 0.35 |
| 400 / 60 | 321 | 10/15 | 13/15 | 15/15 | 0.45 | 0.35 |
| **600 / 100 (used)** | 228 | 10/15 | 13/15 | 15/15 | 0.50 | 0.27 |
| 1000 / 150 | 142 | 10/15 | 12/15 | 13/15 | 0.40 | 0.26 |
| 1500 / 200 | 101 | 10/15 | 13/15 | 14/15 | 0.40 | 0.26 |

600 / 100 tied for the best hit rate and had the widest gap between real and unanswerable questions, which is what makes a fixed cutoff safe. Five small chunks also give the model less text than five large ones, so the settings are not equally generous.

### 3. Answer quality (`python eval_answers.py`, about 6 Gemini requests)

6 answerable questions and 1 unanswerable question were run through the full pipeline and checked against the cited pages.

| Result | Count |
|---|---|
| Correct and grounded in the PDFs | 6/6 |
| Correct but brief (boosting) | 1 of those 6 |
| Unanswerable question correctly rejected | 1/1 |

### Limits of this evaluation

- The test set is small (19 questions), so a difference of one question is noise.
- Most questions were drafted with AI assistance and then checked against the PDFs by the developer.
- The answer review covers 6 answers, was done by the developer with AI assistance, and is a spot check, not an accuracy measure.
- The similarity cutoff (0.35) was chosen on the same questions it is evaluated on.
- Similarity scores depend on the documents and the embedding model. Measure again with `check_scores.py` if either changes.

## Known limitations

- **Sources:** the source list shows the pages that were retrieved, not only the ones Gemini used, so it is often broader than needed.
- **Adding documents:** new PDFs are added through the upload box. Files placed in `data/` after the database was first built are not indexed. Uploading a changed file with the same name is ignored. To rebuild from scratch, delete the `chroma_db/` folder. There is no delete button yet.
- **Scanned PDFs:** OCR is off by default, so pages without text are skipped with a warning. The OCR code is in `ingest.py` (EasyOCR) but is not enabled or listed in `requirements.txt`.
- **Diagrams and charts** are not understood, only the text.
- **Follow-up detection** is keyword based (words such as "it", "that", "more", "again"), so a follow-up without such words may not be rewritten. Each rewrite costs one extra Gemini request.
- **Page requests** match "page N" in the question. With several documents and "Search in" set to all, the pages of every document are combined.
- **Shared data:** there is no login, so all users of one running app share the same documents and database.
- **Upload size:** the maximum upload depends on where the app runs (a 32 MB file was rejected in GitHub Codespaces).
- **API quota:** the Gemini free tier has low daily and per-minute limits that vary by model and change over time. A question costs one request, a follow-up costs two, and questions rejected by the cutoff cost none.
- **Chat history** lives only in the browser session.

## Roadmap

- [x] Multi-PDF ingestion with a persistent vector database
- [x] Answers with sources, and a "not found" response
- [x] Page requests, notes and exercises
- [x] Upload from the browser and a "Search in" filter
- [x] Similarity cutoff to skip the LLM for unrelated questions
- [x] Follow-up questions with chat memory
- [x] Automatic retry and clear Gemini error messages
- [x] Evaluation: retrieval, chunk sizes, answer review
- [ ] Deployment
- [ ] OCR for scanned PDFs
- [ ] Delete a document from the database
- [ ] Show only the pages the model actually used
- [ ] Other file types (Word, PowerPoint)
- [ ] Configurable domain settings (prompts, chunking) per use case
- [ ] Login and document-level access control

## Documents

The PDFs used during development are not included in this repository. Use your own documents.