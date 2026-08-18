import os
import json
import time
import asyncio
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
from app.services.rag_service import retriever_service
from app.services.llm_service import llm_service
from app.services.vector_service import vector_service
from app.services.embedding_service import embedding_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class RAGEvaluator:
    def __init__(self, dataset_path: str = "tests/eval_dataset.json"):
        self.dataset_path = dataset_path
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.docs_dir = self._resolve_eval_docs_dir()
        self.dataset = self._load_dataset()
        logger.info(f"Loaded evaluation dataset with {len(self.dataset)} items from '{dataset_path}' using docs directory '{self.docs_dir}'")

    def _resolve_eval_docs_dir(self) -> str:
        uploads_dir = os.path.join(self.project_root, "data", "uploads")
        if os.path.isdir(uploads_dir):
            return uploads_dir
        return uploads_dir

    def _load_dataset(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if isinstance(dataset, dict) and "questions" in dataset:
            dataset = dataset["questions"]
        if not isinstance(dataset, list):
            raise ValueError(f"Evaluation dataset at '{self.dataset_path}' must be a JSON list of question objects.")
        return dataset

    def setup_eval_documents(self):
        """
        Indexes only the benchmark documents in data/uploads. This ensures the evaluation
        runs against the upload corpus rather than any bundled test docs, and it clears the
        active collection before re-indexing so repeated calls do not duplicate chunks.
        """
        docs_dir = self.docs_dir
        if not os.path.exists(docs_dir):
            logger.warning(f"Evaluation docs directory '{docs_dir}' does not exist.")
            return

        collection_name = "apexguard_rag"
        try:
            vector_service.client.get_collection(collection_name)
            logger.warning(f"Clearing prior vector index '{collection_name}' before re-indexing benchmark docs.")
            vector_service.client.delete_collection(collection_name)
        except Exception:
            pass

        files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
        if not files:
            logger.warning(f"No document files found in '{docs_dir}' for evaluation indexing.")
            return

        from app.services.ingestion_service import DocumentIngestionService, DocumentChunkerService, ChunkingStrategy

        logger.info(f"Indexing {len(files)} real document files from '{docs_dir}' into Qdrant...")
        for filename in files:
            file_path = os.path.join(docs_dir, filename)
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
    def _normalize_metric_text(text: str) -> str:
        return " ".join(text.casefold().split())

    @classmethod
    def _unique_ground_truth_chunks(cls, ground_truth_chunks: List[str]) -> List[str]:
        unique_chunks = []
        seen = set()
        for chunk in ground_truth_chunks:
            normalized = cls._normalize_metric_text(chunk)
            if normalized and normalized not in seen:
                unique_chunks.append(normalized)
                seen.add(normalized)
        return unique_chunks

    @staticmethod
    def _is_match(retrieved_chunk: str, ground_truth_chunk: str) -> bool:
        return ground_truth_chunk in retrieved_chunk or retrieved_chunk in ground_truth_chunk

    @classmethod
    def _matched_ground_truth_indexes(
        cls,
        retrieved_chunks: List[str],
        ground_truth_chunks: List[str],
        k: int
    ) -> set[int]:
        matched_indexes = set()
        seen_retrieved_chunks = set()

        for chunk in retrieved_chunks[:k]:
            normalized_chunk = cls._normalize_metric_text(chunk)
            if not normalized_chunk or normalized_chunk in seen_retrieved_chunks:
                continue

            seen_retrieved_chunks.add(normalized_chunk)
            for idx, gt in enumerate(ground_truth_chunks):
                if idx not in matched_indexes and cls._is_match(normalized_chunk, gt):
                    matched_indexes.add(idx)

        return matched_indexes

    @classmethod
    def calculate_precision_at_k(cls, retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
        top_k = retrieved_chunks[:k]
        if not top_k:
            return 0.0
        unique_ground_truth = cls._unique_ground_truth_chunks(ground_truth_chunks)
        matched_targets = cls._matched_ground_truth_indexes(top_k, unique_ground_truth, k)
        return len(matched_targets) / k

    @classmethod
    def calculate_recall_at_k(cls, retrieved_chunks: List[str], ground_truth_chunks: List[str], k: int) -> float:
        unique_ground_truth = cls._unique_ground_truth_chunks(ground_truth_chunks)
        if not unique_ground_truth:
            return 0.0
        matched_targets = cls._matched_ground_truth_indexes(retrieved_chunks, unique_ground_truth, k)
        return len(matched_targets) / len(unique_ground_truth)

    @classmethod
    def calculate_mrr(cls, retrieved_chunks: List[str], ground_truth_chunks: List[str]) -> float:
        unique_ground_truth = cls._unique_ground_truth_chunks(ground_truth_chunks)
        if not unique_ground_truth:
            return 0.0

        seen_retrieved_chunks = set()
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            normalized_chunk = cls._normalize_metric_text(chunk)
            if not normalized_chunk or normalized_chunk in seen_retrieved_chunks:
                continue

            seen_retrieved_chunks.add(normalized_chunk)
            if any(cls._is_match(normalized_chunk, gt) for gt in unique_ground_truth):
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

    async def evaluate_strategy(
        self,
        strategy_name: str,
        k: int = 3,
        max_questions: Optional[int] = None,
        include_generation: bool = True
    ) -> Dict[str, Any]:
        dataset = self.dataset[:max_questions] if max_questions is not None else self.dataset
        logger.info(
            f"--- Evaluating Strategy: {strategy_name} (K={k}, questions={len(dataset)}, "
            f"include_generation={include_generation}) ---"
        )
        precision_list = []
        recall_list = []
        mrr_list = []
        correctness_list = []
        latencies = []
        item_details = []

        for item in dataset:
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

            generated_answer = ""
            correctness = 0.0
            if include_generation:
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
            "answer_correctness": float(np.mean(correctness_list)) if correctness_list else 0.0,
            "avg_latency_ms": float(np.mean(latencies)),
            "item_details": item_details
        }
        return metrics

    async def run_evaluation(
        self,
        k: int = 3,
        max_questions: Optional[int] = None,
        strategies: Optional[List[str]] = None,
        include_generation: bool = True
    ) -> List[Dict[str, Any]]:
        self.setup_eval_documents()
        strategies = strategies or ["dense", "hybrid", "hybrid_reranked"]
        report_data = []

        for strat in strategies:
            res = await self.evaluate_strategy(
                strat,
                k=k,
                max_questions=max_questions,
                include_generation=include_generation
            )
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
            f.write(f"**Benchmark Dataset Size:** {len(self.dataset)} questions\n")
            f.write(f"**Questions Evaluated:** {max_questions or len(self.dataset)}\n")
            f.write(f"**Answer Generation:** {'enabled' if include_generation else 'disabled'}\n\n")
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
    parser = argparse.ArgumentParser(description="Run the ApexGuard RAG evaluation harness.")
    parser.add_argument("--k", type=int, default=3, help="Retrieved chunks per question.")
    parser.add_argument("--max-questions", type=int, default=5, help="Maximum questions to evaluate.")
    parser.add_argument(
        "--strategy",
        choices=["dense", "hybrid", "hybrid_reranked", "all"],
        default="dense",
        help="Retrieval strategy to evaluate."
    )
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Run Ollama answer generation and semantic answer scoring."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all questions, all strategies, and answer generation."
    )
    args = parser.parse_args()

    evaluator = RAGEvaluator()
    if args.full:
        asyncio.run(evaluator.run_evaluation(k=args.k))
    else:
        strategies = None if args.strategy == "all" else [args.strategy]
        asyncio.run(
            evaluator.run_evaluation(
                k=args.k,
                max_questions=args.max_questions,
                strategies=strategies,
                include_generation=args.include_generation
            )
        )
