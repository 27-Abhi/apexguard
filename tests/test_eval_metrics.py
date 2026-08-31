from tests.run_eval import RAGEvaluator


def test_recall_counts_each_ground_truth_target_once():
    retrieved_chunks = [
        "JWT_SECRET and MINIO_BUCKET are required.",
        "JWT_SECRET and MINIO_BUCKET are required.",
        "JWT_SECRET and MINIO_BUCKET are required.",
        "BANKING_BOT_API_KEY is also required.",
    ]
    ground_truth_chunks = [
        "JWT_SECRET",
        "MINIO_BUCKET",
        "BANKING_BOT_API_KEY",
    ]

    recall = RAGEvaluator.calculate_recall_at_k(retrieved_chunks, ground_truth_chunks, k=10)

    assert recall == 1.0


def test_precision_counts_unique_matched_targets_only():
    retrieved_chunks = [
        "JWT_SECRET and MINIO_BUCKET are required.",
        "JWT_SECRET and MINIO_BUCKET are required.",
        "unrelated chunk",
    ]
    ground_truth_chunks = [
        "JWT_SECRET",
        "MINIO_BUCKET",
    ]

    precision = RAGEvaluator.calculate_precision_at_k(retrieved_chunks, ground_truth_chunks, k=10)

    assert precision == 0.1


def test_mrr_uses_first_relevant_retrieved_rank():
    retrieved_chunks = [
        "unrelated chunk",
        "another unrelated chunk",
        "The app listens on http://0.0.0.0:7867.",
    ]
    ground_truth_chunks = ["http://0.0.0.0:7867"]

    mrr = RAGEvaluator.calculate_mrr(retrieved_chunks, ground_truth_chunks)

    assert mrr == 1 / 3
