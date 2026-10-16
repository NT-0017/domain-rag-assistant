# eval_answers.py
# PURPOSE: Run a few test questions through the FULL pipeline (search + cutoff +
#          Gemini) and save the answers so you can judge them against the PDFs.
#          Uses about 6 Gemini requests.
import time

from eval_retrieval import load_questions, QUESTIONS_FILE
from generator import assistant, GeminiError

# Each entry must match the start of a line in eval_questions.txt
PICKED = [
    "What is the difference between bagging and pasting",
    "How does boosting train its predictors",
    "How was the class imbalance handled in the customer churn project",
    "What should I say when asked how I use AI tools",
    "How should I talk about my NPTEL certification",
    "How is machine learning different from traditional rule-based software",
    "How do I fit a linear regression model",   # no-answer question, costs no quota
]
OUTPUT_FILE = "eval_answers_output.md"


def main():
    items = load_questions(QUESTIONS_FILE)
    chosen = []
    for p in PICKED:
        match = next((i for i in items if i["q"].startswith(p)), None)
        if match is None:
            print(f"Not found in {QUESTIONS_FILE}: {p}")
        else:
            chosen.append(match)

    lines = []
    for n, it in enumerate(chosen, start=1):
        print(f"\n===== Q{n}: {it['q']}")
        try:
            answer, sources = assistant(it["q"])
        except GeminiError as e:
            print("STOPPED:", e)
            lines.append(f"## Q{n}: {it['q']}\n\nSTOPPED: {e}\n")
            break

        if it["none"]:
            expected = "NONE (should be rejected)"
            retrieved = "correct" if not sources else "NOT REJECTED"
        else:
            expected = f"{it['file']} p{sorted(it['pages'])}"
            found = any(f == it["file"] and p in it["pages"] for f, p in sources)
            retrieved = "yes" if found else "NO"

        src_text = ", ".join(f"{f} p{p}" for f, p in sources) or "(none)"
        print(answer)
        print(f"\nExpected: {expected}")
        print(f"Sources shown: {src_text}")
        print(f"Expected page among sources: {retrieved}")

        lines.append(
            f"## Q{n}: {it['q']}\n\nExpected: {expected}\n\n"
            f"Sources shown: {src_text}\n\nExpected page among sources: {retrieved}\n\n"
            f"Answer:\n\n{answer}\n\nYour rating: ____ (Correct / Partly / Wrong)\n"
            f"Grounded in the cited pages: ____ (Yes / No)\n\n---\n"
        )
        time.sleep(4)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()