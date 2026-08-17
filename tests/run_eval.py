import os
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any
from app.services.rag_service import retriever_service
from app.services.llm_service import llm_service
from app.services.vector_service import vector_service
from app.services.embedding_service import embedding_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class RAGEvaluator:
    def __init__(self, dataset_path: str = "tests/eval_dataset.json"):
        self.dataset_path = dataset_path
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        logger.info(f"Loaded evaluation dataset with {len(self.dataset)} items from '{dataset_path}'")

    def setup_eval_documents(self):
        """
        Indexes real benchmark files from data/uploads into Qdrant vector database.
        If data/uploads has files, extracts and ingests them using the DocumentIngestionService.
        """
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
        if not os.path.exists(uploads_dir):
            logger.warning(f"Uploads directory '{uploads_dir}' does not exist.")
            return

        files = [f for f in os.listdir(uploads_dir) if os.path.isfile(os.path.join(uploads_dir, f))]
        if not files:
            logger.warning(f"No document files found in '{uploads_dir}' for evaluation indexing.")
            return

        from app.services.ingestion_service import DocumentIngestionService, DocumentChunkerService, ChunkingStrategy
        
        logger.info(f"Indexing {len(files)} real document files from '{uploads_dir}' into Qdrant...")
        for filename in files:
            file_path = os.path.join(uploads_dir, filename)
            try:
                raw_text = DocumentIngestionService.extract_text_from_file(file_path, filename)
                if not raw_text.strip():
                    continue
                chunks = DocumentChunkerService.chunk_document(raw_text, strategy=ChunkingStrategy.RECURSIVE, chunk_size=500, overlap=50)
                if chunks:
                    embeddings = embedding_service.embed_texts(chunks)
                    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
                    vector_service.insert_documents(chunks=chunks, embeddings=embeddings, metadatas=metadatas)
                    logger.info(f"Ingested {len(chunks)} chunks for document '{filename}' into vector DB.")
            except Exception as e:
                logger.error(f"Failed to ingest eval document '{filename}': {e}")
        logger.info("Evaluation document indexing complete.")

    @staticmethod
    def calculate_precision_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
        top_k = retrieved_chunks[:k]
        if not top_k:
            return 0.0
        relevant_retrieved = sum(1 for chunk in top_k if any(gt in chunk or chunk in gt for gt in ground_truth_chunks))
        return relevant_retrieved / k

    @staticmethod
    def calculate_recall_at_k(retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
        top_k = retrieved_chunks[:k]
        if not ground_truth_chunks:
            return 0.0
        relevant_retrieved = sum(1 for chunk in top_k if any(gt in chunk or chunk in gt for gt in ground_truth_chunks))
        return relevant_retrieved / len(ground_truth_chunks)

    @staticmethod
    def calculate_mrr(retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            if any(gt in chunk or chunk in gt for gt in ground_truth_chunks):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """Calculates cosine similarity between two text strings using FastEmbed embeddings."""
        if not text1 or not text2:
            return 0.0
        v1 = np.array(embedding_service.embed_query(text1))
        v2 = np.array(embedding_service.embed_query(text2))
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def evaluate_strategy(self, strategy_name: str, k: int = 3) -> Dict[str, Any]:
        logger.info(f"--- Evaluating Strategy: {strategy_name} (K={k}) ---")
        precision_list = []
        recall_list = []
        mrr_list = []
        correctness_list = []
        latencies = []
        item_details = []

        for item in self.dataset:
            question = item["question"]
            expected_answer = item["expected_answer"]
            ground_truth_chunks = item.get("ground_truth_chunks", [])
            ground_truth_doc = item.get("ground_truth_doc", "")

            start_time = time.time()
            
            # Execute retrieval based on strategy
            if strategy_name == "dense":
                results = retriever_service.retrieve(query=question, top_k=k)
            elif strategy_name == "hybrid":
                results = retriever_service.retrieve_hybrid(
                    query=question, top_k=k, enable_rerank=False
                )
            elif strategy_name == "hybrid_reranked":
                results = retriever_service.retrieve_hybrid(
                    query=question, top_k=k, enable_rerank=True
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")

            elapsed_ms = (time.time() - start_time) * 1000
            latencies.append(elapsed_ms)

            retrieved_chunk_texts = [res["text"] for res in results]

            # Compute retrieval metrics
            p_k = self.calculate_precision_at_k(retrieved_chunk_texts, ground_truth_chunks, k)
            r_k = self.calculate_recall_at_k(retrieved_chunk_texts, ground_truth_chunks, k)
            mrr = self.calculate_mrr(retrieved_chunk_texts, ground_truth_chunks)

            precision_list.append(p_k)
            recall_list.append(r_k)
            mrr_list.append(mrr)

            # Generate answer and evaluate Answer Correctness (Semantic Similarity)
            context = retriever_service.format_context(results)
            generated_answer = await llm_service.generate_answer(question, context)
            correctness = self.calculate_semantic_similarity(generated_answer, expected_answer)
            correctness_list.append(correctness)

            item_details.append({
                "id": item.get("id"),
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "ground_truth_doc": ground_truth_doc,
                "ground_truth_chunks": ground_truth_chunks,
                "retrieved_chunks": retrieved_chunk_texts,
                "precision_at_k": p_k,
                "recall_at_k": r_k,
                "mrr": mrr,
                "answer_correctness": correctness,
                "latency_ms": elapsed_ms
            })

        metrics = {
            "strategy": strategy_name,
            "precision_at_k": float(np.mean(precision_list)),
            "recall_at_k": float(np.mean(recall_list)),
            "mrr": float(np.mean(mrr_list)),
            "answer_correctness": float(np.mean(correctness_list)),
            "avg_latency_ms": float(np.mean(latencies)),
            "item_details": item_details
        }
        return metrics

    async def run_evaluation(self, k: int = 3) -> List[Dict[str, Any]]:
        self.setup_eval_documents()
        strategies = ["dense", "hybrid", "hybrid_reranked"]
        report_data = []

        for strat in strategies:
            res = await self.evaluate_strategy(strat, k=k)
            report_data.append(res)

        # Generate report output files
        os.makedirs("docs/eval_reports", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        json_path = f"docs/eval_reports/eval_report_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        md_path = f"docs/eval_reports/eval_report_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 📊 ApexGuard Phase 3 — RAG & LLM Evaluation Report\n\n")
            f.write(f"**Generated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Evaluated Strategies:** {', '.join(strategies)}\n")
            f.write(f"**Benchmark Dataset Size:** {len(self.dataset)} questions\n\n")
            f.write(f"## Summary Metrics\n\n")
            f.write(f"| Strategy | Precision@{k} | Recall@{k} | MRR | Answer Correctness | Avg Latency (ms) |\n")
            f.write(f"| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for r in report_data:
                f.write(
                    f"| `{r['strategy']}` | {r['precision_at_k']:.4f} | {r['recall_at_k']:.4f} | "
                    f"{r['mrr']:.4f} | {r['answer_correctness']:.4f} | {r['avg_latency_ms']:.2f} |\n"
                )
            
            f.write("\n---\n## Item-by-Item Detailed Breakdown\n\n")
            for r in report_data:
                strat = r["strategy"]
                f.write(f"### Strategy: `{strat}`\n\n")
                for item in r.get("item_details", []):
                    f.write(f"#### [{item['id']}] Question: {item['question']}\n")
                    f.write(f"- **Expected Answer:** {item['expected_answer']}\n")
                    f.write(f"- **Generated Answer:** {item['generated_answer']}\n")
                    f.write(f"- **Target Doc:** `{item['ground_truth_doc']}`\n")
                    f.write(f"- **Scores:** Precision@{k}: `{item['precision_at_k']:.2f}`, Recall@{k}: `{item['recall_at_k']:.2f}`, MRR: `{item['mrr']:.2f}`, Answer Correctness: `{item['answer_correctness']:.4f}`, Latency: `{item['latency_ms']:.2f}ms`\n")
                    f.write(f"- **Ground Truth Targets:**\n")
                    for gt in item["ground_truth_chunks"]:
                        f.write(f"  - `{gt}`\n")
                    f.write(f"- **Retrieved Chunks:**\n")
                    for idx, chunk in enumerate(item["retrieved_chunks"], start=1):
                        clean_chunk = chunk.replace('\n', ' ')
                        f.write(f"  {idx}. \"{clean_chunk[:150]}...\"\n")
                    f.write("\n")

            f.write("\n---\n*Report automatically generated by `tests.run_eval` harness.*")

        logger.info(f"Evaluation report generated successfully:\n - JSON: {json_path}\n - Markdown: {md_path}")
        return report_data

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    asyncio.run(evaluator.run_evaluation())
