import json
from core.retrieval import retrieve_and_rerank

def recall_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    return len(set(top_k) & set(relevant)) / len(relevant)

def average_precision_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k]):
        if p in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)

# def success_at_k(predicted, relevant, k):
#     return 1.0 if set(predicted[:k]) & set(relevant) else 0.0

def evaluate(eval_data, k=10):
    recall_scores = []
    map_scores = []
    success_scores = []

    for entry in eval_data:
        query = entry["query"]
        relevant_names = entry["relevant_names"]

        print(f"\n🔍 Evaluating: {query}")
        print(f"🎯 Ground truth: {relevant_names}")

        try:
            results = retrieve_and_rerank(query)
            predicted = [r["Test Name"] for r in results]

            print(f"📥 Retrieved: {predicted}")

            recall = recall_at_k(predicted, relevant_names, k)
            ap = average_precision_at_k(predicted, relevant_names, k)
            # hit = success_at_k(predicted, relevant_names, k)

            recall_scores.append(recall)
            map_scores.append(ap)
            # success_scores.append(hit)

            print(f"✅ Recall@{k}: {recall:.4f} | AP@{k}: {ap:.4f} ")

        except Exception as e:
            print(f"❌ Error for query '{query}': {e}")
            recall_scores.append(0.0)
            map_scores.append(0.0)
            # success_scores.append(0.0)

    return {
        f"Mean Recall@{k}": round(sum(recall_scores) / len(recall_scores), 4),
        f"MAP@{k}": round(sum(map_scores) / len(map_scores), 4),
        # f"Success@{k}": round(sum(success_scores) / len(success_scores), 4)
    }

if __name__ == "__main__":
    with open("eval_data.json") as f:
        eval_data = json.load(f)

    results = evaluate(eval_data, k=10)

    print("Final Evaluation Metrics:")
    for key, value in results.items():
        print(f"{key}: {value}")
