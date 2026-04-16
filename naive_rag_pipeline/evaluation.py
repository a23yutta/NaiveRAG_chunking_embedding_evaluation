from rank_eval import evaluate

# ==============================
#       NODE ID EXTRACTION
# ==============================
def _get_node_id(node):
    """
    Extract identifier from a node object.

    Supports multiple formats:
    - Raw string IDs
    - LlamaIndex Document objects (via metadata)
    - Generic objects with node_id attribute
    """
    # Case 1: already a string ID
    if isinstance(node, str):
        return node

    # Case 2: LlamaIndex-style node with metadata
    if hasattr(node, "metadata"):
        meta = node.metadata

        # Prefer chunk_id if available (most stable for retrieval evaluation)
        if "chunk_id" in meta:
            return meta["chunk_id"]

    # Case 3: fallback to generic node_id attribute
    if hasattr(node, "node_id"):
        return node.node_id

    # If none of the expected formats match, fail explicitly
    raise ValueError(f"Unsupported node type: {type(node)}")

# ==============================
# FORMAT DATA FOR RANK EVALUATION LIBRARY
# ==============================
def _to_rank_eval(retrieved, ground_truth, query_id="q1"):
    """
    Convert retrieval results into rank_eval format:

    qrels (relevance judgments):
        query -> {doc_id: relevance}

    run (system output ranking):
        query -> {doc_id: score}
    """
    # Ground truth relevance labels (binary relevance = 1)
    qrels = {
        query_id: {
            str(_get_node_id(node)): 1
            for node in ground_truth
        }
    }

    # Model ranking scores (higher = more relevant)
    run = {
        query_id: {
            str(_get_node_id(node)): float(len(retrieved) - i)
            for i, node in enumerate(retrieved)
        }
    }

    return qrels, run

# ==============================
# MAIN EVALUATION FUNCTION
# ==============================
def evaluate_retrieval(retrieved, ground_truth, k=5, query_id="q1"):
    """
    Compute retrieval metrics:
    - Recall@k
    - MRR
    - NDCG@k

    Uses rank_eval library under the hood.
    """
    # Convert data into evaluation format
    qrels, run = _to_rank_eval(retrieved, ground_truth, query_id)

    # Compute ranking metrics
    results = evaluate(
        qrels,
        run,
        metrics=[
            f"recall@{k}",
            "mrr",
            f"ndcg@{k}"
        ]
    )
    # Return clean metric dictionary 
    return {
        "recall@k": results[f"recall@{k}"],
        "mrr": results["mrr"],
        "ndcg@k": results[f"ndcg@{k}"],
    }