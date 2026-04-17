import pandas as pd
from pathlib import Path

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
def run_pipeline(input_dir: Path, output_file: Path):
    files = sorted(input_dir.glob("experiment_results_*.csv"))

    if not files:
        raise ValueError(f"No result files found in {input_dir}")

    print(f"Found {len(files)} run files")

    dfs = []

    for file in files:
        print(f"Loading {file.name}")
        df = load_csv(file)
        dfs.append(df)

    # ---------------------------------------------------------------
    # STEP 1: Merge all experiment results runs into one big dataset
    # ---------------------------------------------------------------
    all_data = pd.concat(dfs, ignore_index=True)

    # --------------------------------------------
    # STEP 2: Compute mean across all runs
    # > For each (chunking, embedding, query_id) --> Average metrics over all experiment runs
    # --------------------------------------------
    final = (
        all_data.groupby(["chunking", "embedding", "query_id"])
        .agg({
            "recall@k": "mean",
            "mrr": "mean",
            "ndcg@k": "mean"
        })
        .reset_index()
        .sort_values(["chunking", "embedding", "query_id"])
    )

    # --------------------------------------------
    # STEP 3: Fix sorting order (NOT alphabetical)
    # > Ensure correct experimental structure order
    # --------------------------------------------
    chunking_order = ["sentence", "recursive", "token"]
    embedding_order = ["openai-small", "openai-large"]

    # Convert to categorical so pandas respects custom order
    final["chunking"] = pd.Categorical(
        final["chunking"],
        categories=chunking_order,
        ordered=True
    )

    final["embedding"] = pd.Categorical(
        final["embedding"],
        categories=embedding_order,
        ordered=True
    )

    # Final sort using experimental hierarchy
    final = final.sort_values(["chunking", "embedding", "query_id"])

    # Save final aggregated dataset
    save_csv(final, output_file)

    print("\nSaved to:", output_file)

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent

    input_dir = BASE / "data" / "results" / "metrics"
    output_file = BASE / "data" / "results" / "analysis" / "final_results.csv"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_dir, output_file)