import sys
from pathlib import Path
import json
from llama_index.core import Document

BASE_DIR = Path(__file__).resolve().parents[1]  # project root

# ==============================
# Predefined chunk file locations
# > Maps chunking strategy → corresponding JSON file
# ==============================
CHUNKING_FILES = {
    "recursive": BASE_DIR / "data/chunks/recursive.json",
    "sentence": BASE_DIR / "data/chunks/sentence.json",
    "token": BASE_DIR / "data/chunks/token.json",
}

# =======================================================
# Load chunk file and convert to LlamaIndex Documents
# =======================================================
def load_chunk_json(path):
    # Validate file exists before reading
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    # Load JSON file
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {path}")

    # Ensure expected structure (list of chunks)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of chunks in {path}")

    documents = []

    # Required metadata fields for each chunk
    required_keys = {"article_id", "paragraph_id", "subpoint_ids"}

    # ===========================================
    # Convert raw JSON chunks → Document objects
    # ===========================================
    for item in data:

        # Ensure expected keys exist in each entry
        if "chunk" not in item or "metadata" not in item:
            raise ValueError(...)

        meta = item["metadata"]

        # Validate required metadata fields
        if not required_keys.issubset(meta):
            raise ValueError(f"Missing metadata keys: {meta}")
        
        # Create LlamaIndex Document object
        documents.append(
            Document(
                text=item["chunk"],
                metadata=item["metadata"]
            )
        )

    return documents