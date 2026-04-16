from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

# ==============================
#       SORTING FUNCTION
# ==============================
def sort_key(x):
    """
    Define sorting logic for retrieval results.

    Sorting priority:
    1. Higher similarity score first
    2. Stable ordering using chunk_id (for deterministic results)

    This ensures reproducibility across runs when scores are equal.
    """
    # Use 0.0 if score is missing 
    score = x.score if x.score is not None else 0.0

    # Fallback to empty string if chunk_id is missing
    chunk_id = x.node.metadata.get("chunk_id", "")

    # Negative score for descending sort
    return (-score, chunk_id)

# ==============================
#           RETRIEVAL 
# ==============================
def retrieve(index, query, top_k=5):
    """
    Retrieve top-k most relevant chunks for a query.

    Steps:
    1. Convert index into a retriever
    2. Perform similarity search
    3. Sort results for deterministic ordering

    Returns:
    - List of retrieved nodes (with scores + metadata)
    """
    # Create retriever from index
    # similarity_top_k controls how many results are returned
    retriever = index.as_retriever(similarity_top_k=top_k)

    # Perform retrieval (vector similarity search)
    results = retriever.retrieve(query)

    # Return sorted results to ensure consistent ordering
    return sorted(results, key=sort_key)
