import json
import pandas as pd
import argparse
from pathlib import Path
from dataclasses import dataclass

from index import build_index
from retrieval import retrieve
from ground_truth import get_ground_truth_nodes
from evaluation import evaluate_retrieval

# Available experiment dimensions
CHUNKING_STRATEGIES = ["sentence", "recursive", "token"]
EMBEDDING_MODELS = ["openai-small", "openai-large"]

# Number of retrieved results (top 5 relevant chunks) per query
TOP_K = 5

# ==============================
#   EXPERIMENT CONFIG OBJECT
# ==============================
@dataclass
class ExperimentConfig:
    # Stores configuration for a single experiment
    chunking: str
    embedding: str
    top_k: int = TOP_K

# ==============================
#          I/O UTILITIES
# ==============================
def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==================================
# CORE PIPELINE (SINGLE EXPERIMENT)
# ==================================
def run_single_experiment(config: ExperimentConfig, dataset, log_dir: Path, run_id: int):
    """
    Runs ONE experiment (the selected one) configuration over the full dataset.

    Steps:
    1. Build index
    2. Run retrieval for each query
    3. Evaluate retrieval performance
    4. Save detailed retrieval logs

    Returns:
    - List of per-query evaluation results
    """
    # Unique name for this experiment setup
    config_key = f"{config.chunking}_{config.embedding}".replace("/", "_")

    # Path to store retrieval logs for this run
    log_path = log_dir / f"run_{run_id}" / f"{config_key}.json"

    print(f"\nRunning: {config.chunking} + {config.embedding}")

    # ------------------------
    # 1. BUILD INDEX
    # ------------------------
    index, documents = build_index(
        config.chunking,
        config.embedding
    )

    # Sanity check: ensure documents exist
    assert len(documents) > 0

    # Validate uniqueness of chunk IDs (important for evaluation correctness)
    seen = set()
    for doc in documents:
        key = doc.metadata["chunk_id"]

        if key in seen:
            print("DUPLICATE FOUND:", key)
        else:
            seen.add(key)

    print("Validation passed")

    # Store evaluation metrics + logs
    results = []
    retrieval_logs = {}

    # ------------------------
    # 2. EVALUATE DATASET
    # ------------------------
    for i, item in enumerate(dataset):
        query = item["question"]
        evidence = item["evidence"]

        # Retrieve top-k documents
        retrieved = retrieve(index, query, config.top_k)

        # Get ground truth nodes (correct answers)
        ground_truth_nodes = get_ground_truth_nodes(documents, evidence)

        # Compute evaluation metrics
        score = evaluate_retrieval(retrieved, ground_truth_nodes, k=config.top_k)

        # Store per-query metrics
        results.append({
            "chunking": config.chunking,
            "embedding": config.embedding,
            "query_id": i,
            "recall@k": score["recall@k"],
            "mrr": score["mrr"],
            "ndcg@k": score["ndcg@k"]
        })

        # Store detailed retrieval logs (for debugging / analysis)
        retrieval_logs[i] = {
            "query_id": i,
            "question": query,
            "retrieved": [
                {
                    "text": r.node.get_content(),
                    "score": float(r.score) if r.score is not None else None,
                    "metadata": r.node.metadata
                }
                for r in retrieved
            ],
            "ground_truth": [
                {
                    "text": d.get_content(),
                    "metadata": d.metadata
                }
                for d in ground_truth_nodes
            ]
        }
    # Save logs for this experiment
    save_json(retrieval_logs, log_path)
    print(f"Saved retrieval logs to: {log_path}")

    return results


# ==================================
# MAIN PIPELINE (MULTI EXPERIMENT)
# ==================================
def run_pipeline(input_qa_dataset: Path, output_results_file: Path, chunking=None, embedding=None, all_runs=False, run_id=1):
    """
    Runs either:
    - A single experiment (specific chunking + embedding), OR
    - A grid of experiments (all combinations)

    Also handles saving final results as CSV.
    """
    BASE = Path(__file__).resolve().parent.parent

    # Load dataset
    dataset = load_dataset(input_qa_dataset)

    # Directory for retrieval logs
    log_dir = BASE / "data" / "results" / "retrieval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"run_{run_id}").mkdir(parents=True, exist_ok=True)

    results = []

    # ------------------------
    #      SINGLE RUN MODE
    # ------------------------
    if not all_runs:
        config = ExperimentConfig(
            chunking=chunking,
            embedding=embedding
        )
        results = run_single_experiment(config, dataset, log_dir, run_id)

    # ------------------------
    #       GRID MODE
    # ------------------------
    else:
        configs = [
            ExperimentConfig(c, e)
            for c in CHUNKING_STRATEGIES
            for e in EMBEDDING_MODELS
        ]

        for config in configs:
            results.extend(
                run_single_experiment(config, dataset, log_dir, run_id)
            )

    # Save final results as CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_results_file, index=False)

    print("\nSaved per-query results to:", output_results_file)


# ==============================
#       CLI ENTRYPOINT
# ==============================
# CLI usage (commands) examples

# Single run (default)
# python run_experiments.py --chunking sentence --embedding openai-small

# 5 repeated runs
# python run_experiments.py --chunking sentence --embedding openai-small --runs 5

# Full grid × 5 runs (3 chunking × 2 embedding × 5 runs = 30 experiments)
# python run_experiments.py --all --runs 5

if __name__ == "__main__":

    BASE = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser()

    # Choose specific experiment
    parser.add_argument("--chunking", type=str, default=None)
    parser.add_argument("--embedding", type=str, default=None)

    # Run all combinations
    parser.add_argument("--all", action="store_true")

    # Number of repeated runs (for stability experiments)
    parser.add_argument("--runs", type=int, default=1)

    args = parser.parse_args()

    # Input dataset
    input_qa_dataset_file = BASE / "data" / "queries" / "final_rewritten_queries.json"

    # Ensure output directories exist
    (BASE / "data" / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (BASE / "data" / "results" / "retrieval_logs").mkdir(parents=True, exist_ok=True)

    # Validate input arguments
    if not args.all and (args.chunking is None or args.embedding is None):
        raise ValueError("Provide --chunking and --embedding OR use --all")

    # ------------------------
    # MULTIPLE RUNS LOOP
    # ------------------------
    for run_id in range(1, args.runs + 1):
        print(f"\n================ RUN {run_id} ================\n")

        # Output file per run (results)
        output_results_file = BASE / "data" / "results" / "metrics" / f"experiment_results_{run_id}.csv"

        run_pipeline(
            input_qa_dataset_file,
            output_results_file,
            chunking=args.chunking,
            embedding=args.embedding,
            all_runs=args.all,
            run_id=run_id
        )