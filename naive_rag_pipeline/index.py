import os
from pathlib import Path
from llama_index.core import VectorStoreIndex

from chunking_config import CHUNKING_FILES, load_chunk_json
from embedding_config import get_embedding_model

os.environ["OPENAI_API_KEY"] = "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"

# ==============================
#       INDEX BUILDING
# ==============================
def build_index(chunking_strategy, embedding_model_name):
    """
    Build a vector index for retrieval.

    Steps:
    1. Load pre-chunked documents
    2. Initialize embedding model
    3. Create vector index from documents

    Returns:
    - index: VectorStoreIndex (used for retrieval)
    - documents: original documents (used for evaluation / ground truth)
    """

    # Validate chunking strategy
    if chunking_strategy not in CHUNKING_FILES:
        raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")

    # Resolve path to chunked dataset
    chunk_path = CHUNKING_FILES[chunking_strategy]

    # ------------------------
    # Load chunked documents
    # ------------------------
    # Each document contains:
    # - text (chunk content)
    # - metadata (article, paragraph, subpoint, chunk_id, etc.)
    documents = load_chunk_json(chunk_path)

    # ------------------------
    # Initialize embedding model
    # ------------------------
    # Supports different embedding backends (OpenAI, HuggingFace, etc.)
    embed_model = get_embedding_model(embedding_model_name)

    # ------------------------
    # Build vector index
    # ------------------------
    # Converts documents into vector embeddings and stores them
    # for similarity-based retrieval
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model
    )

    return index, documents