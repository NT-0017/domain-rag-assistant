# check_scores.py
# PURPOSE: Print the best similarity score for related and unrelated questions,
#          so we can choose a sensible cutoff. Uses no Gemini requests.
from store import embed_model, get_collection

collection = get_collection()

related = [
    "What is hard voting?",
    "What is out-of-bag evaluation?",
    "What is supervised learning?",
    "What challenges were faced in the age and gender detection project?",
    "Explain boosting",
]
unrelated = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the cricket world cup?",
    "What is the weather today?",
    "Explain the rules of chess",
]


def best_score(question):
    q_emb = embed_model.encode(question).tolist()
    result = collection.query(query_embeddings=[q_emb], n_results=1)
    return 1 - result["distances"][0][0]


print("\nRELATED questions (should be answerable):")
for q in related:
    print(f"  {best_score(q):.2f}  |  {q}")

print("\nUNRELATED questions (should be rejected):")
for q in unrelated:
    print(f"  {best_score(q):.2f}  |  {q}")