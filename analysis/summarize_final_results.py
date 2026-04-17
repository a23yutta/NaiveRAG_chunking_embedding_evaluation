import pandas as pd
from pathlib import Path
import numpy as np

# =========================
#       IO UTILITIES
# =========================
def load_csv(path: Path):
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

# =========================
#       MAIN PIPELINE
# =========================
def run_pipeline(input_file: Path, output_file: Path):
    df = load_csv(input_file)
    print("Loaded:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # --------------------------------------------
    #      STEP 1: Group by experiment config
    # --------------------------------------------
    grouped = df.groupby(["chunking", "embedding"])

    summary = grouped.agg(
        recall_mean=("recall@k", "mean"),
        recall_std=("recall@k", "std"),

        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std"),

        ndcg_mean=("ndcg@k", "mean"),
        ndcg_std=("ndcg@k", "std"),
    ).reset_index()

    # --------------------------------------------
    #    Fix sorting order (NOT alphabetical)
    # --------------------------------------------
    chunking_order = ["sentence", "recursive", "token"]
    embedding_order = ["openai-small", "openai-large"]

    # Convert to categorical so pandas respects custom order
    summary["chunking"] = pd.Categorical(
        summary["chunking"],
        categories=chunking_order,
        ordered=True
    )

    summary["embedding"] = pd.Categorical(
        summary["embedding"],
        categories=embedding_order,
        ordered=True
    )

    # Final sort using experimental structure
    summary = summary.sort_values(["chunking", "embedding"])

    save_csv(summary, output_file)

    print("\nSaved to:", output_file)

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent

    input_file = BASE / "data" / "results" / "analysis" / "final_results.csv"
    output_file = BASE / "data" / "results" / "analysis" / "final_results_summary.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_file, output_file)