# generator.py
# PURPOSE: Gemini connection plus the question-answering logic:
#          ask (normal questions), smart_ask (page requests),
#          assistant (router for notes / exercises / everything else).
import os
import re
import time
from google.genai import errors as genai_errors

from google import genai

from store import embed_model, get_collection

MODEL_NAME = "gemini-3.5-flash-lite"  # change here if you switch models
NOT_FOUND = "i could not find this in the documents"
REWRITE_ENABLED = os.environ.get("ENABLE_REWRITE", "0") == "1"
MIN_SCORE = 0.35   # best match below this means the documents don't cover the question

_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it as a Codespaces secret "
                       "(or export it in your terminal) and restart.")
gemini = genai.Client(api_key=_api_key)
collection = get_collection()


class GeminiError(Exception):
    """An error whose message is safe to show to the user."""


RETRY_WAITS = [3, 8]   # seconds to wait before retry 1 and retry 2


def _generate(prompt):
    """Call Gemini. Retries only when the server is busy (500/503)."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            text = gemini.models.generate_content(
                model=MODEL_NAME, contents=prompt).text
            if not text:
                raise GeminiError("Gemini returned an empty answer. "
                                  "Try rephrasing your question.")
            return text
        except genai_errors.APIError as e:
            code = getattr(e, "code", None)
            if code == 429:
                raise GeminiError("The Gemini free quota is used up (daily or "
                                  "per-minute limit). Please wait and try again "
                                  "later.") from e
            if code in (500, 503):
                if attempt < len(RETRY_WAITS):
                    wait = RETRY_WAITS[attempt]
                    print(f"Gemini busy ({code}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise GeminiError("Gemini is busy right now. I tried 3 times. "
                                  "Please try again in a minute.") from e
            raise GeminiError(f"Gemini returned an error ({code}). "
                              f"Details: {e}") from e

def _unique_sources(metas):
    return sorted({(m["file"], m["page"]) for m in metas})

FOLLOW_UP_WORDS = re.compile(
    r"\b(it|its|that|those|these|they|them|their|more|simpler|shorter|"
    r"again|previous|above|same|another|elaborate|continue)\b"
    r"|^(and|also|but|so)\b|what about|how about",
    re.IGNORECASE,
)


def rewrite_question(question, history):
    """Turn a follow-up into a standalone question using recent chat messages.
    Costs one Gemini request, and only when the message looks like a follow-up."""
    if not history or not FOLLOW_UP_WORDS.search(question):
        return question

    recent = history[-4:]   # last two exchanges
    chat_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}" for m in recent
    )
    prompt = f"""Rewrite the user's latest message as one standalone question or request
that can be understood without the conversation. Keep the original intent, any page
numbers, and any style request (such as "as if I am 5" or "in simpler words").
If it is already standalone, return it unchanged.
Return ONLY the rewritten text.

CONVERSATION:
{chat_text}

LATEST MESSAGE: {question}

STANDALONE VERSION:"""
    rewritten = _generate(prompt).strip()
    print("REWRITTEN:", repr(rewritten))
    return rewritten or question # if Gemini is busy or over quota, use the original

def _finish(answer, metas):
    """Return (answer, sources). Sources are dropped if nothing was found."""
    if NOT_FOUND in answer.lower():
        return answer, []
    return answer, _unique_sources(metas)

NOT_FOUND_MESSAGE = ("I could not find this in the documents. Try naming the topic, "
                     "asking a full question, or choosing one document in Search in.")

STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "does", "did", "can",
    "could", "would", "should", "will", "has", "have", "had", "explain", "describe",
    "tell", "give", "show", "about", "from", "into", "than", "then", "them", "they",
    "their", "there", "your", "you", "please", "summarize", "summary", "mean", "means",
    "define", "meaning", "paper", "document", "pdf", "section",
}
KEYWORD_SLOTS = 2   # how many keyword-matched chunks to add


def _words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _stem(word):
    return word if len(word) <= 5 else word[:5]


def _key_terms(question):
    return {_stem(w) for w in _words(question) if len(w) > 2 and w not in STOPWORDS}


def retrieve(question, k=5, file=None):
    """Meaning search plus keyword search. Returns (docs, metas), or None if nothing
    in the documents matches well enough."""
    where = {"file": file} if file else None
    q_emb = embed_model.encode(question).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, where=where)
    if not res["ids"][0]:
        return None
    best = 1 - res["distances"][0][0]
    chosen = {}   # chunk id -> (text, metadata), in priority order

    # 1. Chunks that contain ALL the key words of the question
    terms = _key_terms(question)
    strong = False
    if terms:
        everything = collection.get(where=where)
        scored = []
        for cid, d, m in zip(everything["ids"], everything["documents"],
                             everything["metadatas"]):
            counts = {}
            for w in _words(d):
                s = _stem(w)
                if s in terms:
                    counts[s] = counts.get(s, 0) + 1
            if counts:
                scored.append((len(counts) / len(terms), sum(counts.values()), cid, d, m))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for frac, _, cid, d, m in scored[:KEYWORD_SLOTS]:
            if frac == 1.0:
                strong = True
                chosen[cid] = (d, m)

    # 2. For a keyword match, also take the chunk after it (sections continue)
    if strong:
        wanted = []
        for cid in list(chosen):
            match = re.match(r"^(.*)_p(\d+)_c(\d+)$", cid)
            if match:
                base, page, i = match.group(1), int(match.group(2)), int(match.group(3))
                wanted += [f"{base}_p{page}_c{i + 1}", f"{base}_p{page + 1}_c0"]
        if wanted:
            extra = collection.get(ids=wanted)
            for cid, d, m in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                chosen.setdefault(cid, (d, m))

    # 3. The meaning-search results, only if they pass the cutoff (or keywords matched)
    if strong or best >= MIN_SCORE:
        for cid, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
            chosen.setdefault(cid, (d, m))

    if not chosen:
        return None
    items = list(chosen.values())[: k + 4]
    return [d for d, _ in items], [m for _, m in items]

SUMMARY_WORDS = re.compile(
    r"\b(summar\w*|overview|tl;?dr|main (points|ideas)|key (points|takeaways)|"
    r"what is (this|the) (paper|document|pdf|file|chapter) about)\b", re.IGNORECASE)
MAX_SUMMARY_CHUNKS = 20


def _document_overview(file):
    """Chunks of one document in reading order, spread evenly if there are many."""
    data = collection.get(where={"file": file})
    items = []
    for cid, d, m in zip(data["ids"], data["documents"], data["metadatas"]):
        pos = re.search(r"_p(\d+)_c(\d+)$", cid)
        if pos:
            items.append(((int(pos.group(1)), int(pos.group(2))), d, m))
    items.sort(key=lambda x: x[0])
    if len(items) > MAX_SUMMARY_CHUNKS:
        step = len(items) / MAX_SUMMARY_CHUNKS
        items = [items[int(i * step)] for i in range(MAX_SUMMARY_CHUNKS)]
    return [d for _, d, _ in items], [m for _, _, m in items]


def _retrieve_with_rewrite(question, k, file):
    """Normal retrieval. If a SHORT message that is not a question fails, ask Gemini
    to turn it into a full question and try once more (costs 1 extra request)."""
    found = retrieve(question, k, file)
    if (found is not None or not REWRITE_ENABLED or len(question.split()) > 6
            or question.strip().endswith("?")):
        return found
    rewritten = _generate(
        "Rewrite this short message as one clear, complete question about a document. "
        "Keep every key word. Return ONLY the question.\n\nMESSAGE: " + question).strip()
    print("RETRY WITH:", repr(rewritten))
    return retrieve(rewritten, k, file) if rewritten else None

def ask(question, k=5, file=None):
    """Meaning-based search, then Gemini writes the answer."""
    found = _retrieve_with_rewrite(question, k, file)
    if found is None:
        return NOT_FOUND_MESSAGE, []
    docs, metas = found

    context = "\n\n".join(
        f"[Source: {m['file']}, page {m['page']}]\n{d}" for d, m in zip(docs, metas)
    )
    prompt = f"""You are a helpful study assistant.
Use ONLY the context below to answer the question.
Understand the question and explain the answer clearly in your own words.
Do not copy the text word for word. Some text comes from OCR and may contain
small errors, so correct obvious mistakes.
If the context does not contain the answer, say: "I could not find this in the documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    return _finish(_generate(prompt), metas)


def smart_ask(question, file=None):
    """If the question names a page, fetch that page directly. Else use ask()."""
    match = re.search(r"page\s*(?:number\s*)?(\d+)", question, re.IGNORECASE)
    if not match:
        return ask(question, file=file)

    page_no = int(match.group(1))
    where = {"page": page_no}
    if file:
        where = {"$and": [{"page": page_no}, {"file": file}]}

    found = collection.get(where=where)
    if not found["documents"]:
        where_txt = f" in {file}" if file else ""
        return f"No content found for page {page_no}{where_txt}.", []

    # Put the chunks back in reading order (c0, c1, c2...)
    items = sorted(
        zip(found["ids"], found["documents"], found["metadatas"]),
        key=lambda x: (x[2]["file"], int(re.search(r"_c(\d+)$", x[0]).group(1))),
    )
    metas = [m for _, _, m in items]
    files_used = sorted({m["file"] for m in metas})
    note = ""
    if len(files_used) > 1:
        note = (f"ℹ️ Page {page_no} exists in {len(files_used)} documents "
                f"({', '.join(files_used)}). Use **Search in** in the sidebar "
                f"to pick one.\n\n")

    context = "\n\n".join(
        f"[Source: {m['file']}, page {m['page']}]\n{d}" for _, d, m in items
    )
    prompt = f"""You are a helpful study assistant.
Use ONLY the page content below. Follow the user's request exactly
(for example, explain simply, summarize, or make questions).
Write in your own words. Some text comes from OCR and may contain small errors,
so correct obvious mistakes.

PAGE CONTENT:
{context}

REQUEST: {question}

RESPONSE:"""
    answer = _generate(prompt)
    return note + answer, _unique_sources(metas)


def assistant(question, file=None, k=10, history=None):
    """Router: exercises and notes use more chunks and a special prompt.
    Everything else goes to smart_ask()."""
    question = rewrite_question(question, history)
    q = question.lower()
    docs = metas = None

    if re.search(r"true\s*/?\s*(or\s*)?false|quiz|mcq|multiple choice|exercise|practice questions", q):
        instruction = ("Create practice questions on this topic. Unless the request gives a number, "
                       "write 5 questions. If the request says True/False, write True/False "
                       "statements; if it says MCQ or multiple choice, write 4 options each. "
                       "Put an ANSWER KEY with a one-line explanation at the end.")
    elif re.search(r"\bnotes\b|revision|study guide|cheat sheet", q):
        instruction = ("Write clear study notes on this topic. Use short headings and bullet points, "
                       "define key terms simply, and end with a 'Key takeaways' list.")
    
    elif SUMMARY_WORDS.search(q) and not re.search(r"page\s*\d+", q):
        target = file
        if target is None:
            names = sorted({m["file"] for m in
                            collection.get(include=["metadatas"])["metadatas"]})
            if len(names) != 1:
                return ("Which document should I summarize? Choose one in **Search in** "
                        "in the sidebar, then ask again.", [])
            target = names[0]
        docs, metas = _document_overview(target)
        if not docs:
            return NOT_FOUND_MESSAGE, []
        instruction = ("Write a clear summary of this document: first its purpose in two "
                       "sentences, then its main sections or points as short bullets, then "
                       "the conclusion if there is one. The text is a sample of excerpts "
                       "in reading order, so do not claim to cover parts that are not shown.")
    else:
        return smart_ask(question, file=file)

    if docs is None:
        found = _retrieve_with_rewrite(question, k, file)
        if found is None:
            return NOT_FOUND_MESSAGE, []
        docs, metas = found

    context = "\n\n".join(
        f"[Source: {m['file']}, page {m['page']}]\n{d}" for d, m in zip(docs, metas)
    )
    prompt = f"""You are a helpful study assistant.
Use ONLY the context below as the source of facts.
{instruction}
Write in your own words; do not copy text word for word. Some text comes from OCR
and may contain small errors, so correct obvious mistakes.
If the context does not contain the topic, say: "I could not find this in the documents."

CONTEXT:
{context}

REQUEST: {question}

RESPONSE:"""
    return _finish(_generate(prompt), metas)


if __name__ == "__main__":
    answer, sources = assistant("What is hard voting?")
    print(answer)
    if sources:
        print("\nSources:")
        for f, p in sources:
            print(f"- {f}, page {p}")
