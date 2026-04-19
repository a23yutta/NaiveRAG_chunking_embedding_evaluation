from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

# =========================
#       IO UTILITIES
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

# ====================================
#      Summary of final_results.py
# ====================================
def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Compute mean and standard deviation for all combination of experiments
    summary = df.groupby(["chunking", "embedding"]).agg(
        recall_mean=("recall@k", "mean"),
        recall_std=("recall@k", "std"),
        mrr_mean=("mrr", "mean"),
        mrr_std=("mrr", "std"),
        ndcg_mean=("ndcg@k", "mean"),
        ndcg_std=("ndcg@k", "std")
    ).reset_index()
    # --------------------------------------
    # Structure ordering (not alphabetical)
    # --------------------------------------
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

    return summary

# ===============================
#      Heatmap visualization
# ===============================
def create_heatmaps(summary: pd.DataFrame, output_dir: Path):
    """
    Create heatmaps for each retrieval metric.

    These visualise aggregated performance across:
    - chunking strategies
    - embedding models
    """
    pivot_recall = summary.pivot(index="chunking", columns="embedding", values="recall_mean")
    pivot_mrr = summary.pivot(index="chunking", columns="embedding", values="mrr_mean")
    pivot_ndcg = summary.pivot(index="chunking", columns="embedding", values="ndcg_mean")

    # Recall heatmap
    plt.figure()
    sns.heatmap(pivot_recall, annot=True, fmt=".3f")
    plt.title("Recall@k Heatmap")
    plt.savefig(output_dir / "heatmap_recall.png")
    plt.close()

    # MRR heatmap
    plt.figure()
    sns.heatmap(pivot_mrr, annot=True, fmt=".3f")
    plt.title("MRR Heatmap")
    plt.savefig(output_dir / "heatmap_mrr.png")
    plt.close()

    # nDCG heatmap
    plt.figure()
    sns.heatmap(pivot_ndcg, annot=True, fmt=".3f")
    plt.title("nDCG Heatmap")
    plt.savefig(output_dir / "heatmap_ndcg.png")
    plt.close()

# ===============================
#      Interaction plots
# ===============================
def create_interaction_plots(summary: pd.DataFrame, output_dir: Path):
    """
    Interaction plots showing how chunking and embedding interact
    across different retrieval metrics.

    These are exploratory visualisations (not statistical inference).
    """
    metrics = [
        ("recall_mean", "Recall@k"),
        ("mrr_mean", "MRR"),
        ("ndcg_mean", "nDCG@k")
    ]

    for col, title in metrics:
        plt.figure()

        sns.pointplot(
            data=summary,
            x="chunking",
            y=col,
            hue="embedding"
        )

        plt.title(f"Interaction Plot ({title})")
        plt.ylabel(f"Mean {title}")

        filename = f"interaction_plot_{col.replace('_mean','')}.png"
        plt.savefig(output_dir / filename)
        plt.close()

# ===============================
# Statistical inference (primary results)
# ===============================
def run_two_way_anova_ndcg_repeated_measures(df: pd.DataFrame, output_dir: Path):
    """
    Repeated-measures two-way ANOVA on nDCG@k (PRIMARY METRIC ONLY).

    This tests the hypotheses:
    H1: Embedding effect
    H2: Chunking effect
    H3: Interaction effect

    and its corresponding research question
    """

    # Two way ANOVA repeated measures
    anova = pg.rm_anova(
        dv="ndcg@k",
        within=["chunking", "embedding"],
        subject="query_id",
        data=df,
        detailed=True
    )

    output_path = output_dir / "anova_ndcg_results.csv"
    anova.to_csv(output_path, index=False)

    with open(output_dir / "anova_ndcg_results.txt", "w") as f:
        f.write(anova.to_string())

    return anova

# =========================
#       MAIN PIPELINE
# =========================
def run_pipeline(input_file: Path, output_dir: Path):
    """
    Full experimental analysis pipeline:

    1. Load raw retrieval results
    2. Compute aggregated statistics for visualization (Summary table)
    3. Generate heatmaps
    4. Generate interaction plots
    5. Run statistical inference (nDCG only)
    """
    df = load_csv(input_file)
    summary = compute_summary(df)
    save_csv(summary, output_dir / "summary.csv")
    create_heatmaps(summary, output_dir)
    create_interaction_plots(summary, output_dir)
    run_two_way_anova_ndcg_repeated_measures(df, output_dir)


if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent

    input_file = BASE / "data" / "results" / "analysis" / "final_results.csv"
    output_dir = BASE / "data" / "results" / "analysis" / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_file, output_dir)