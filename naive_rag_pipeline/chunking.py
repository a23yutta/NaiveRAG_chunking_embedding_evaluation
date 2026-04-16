import json
from pathlib import Path
from dataclasses import dataclass
from llama_index.core import Document
from chonkie import TokenChunker, SentenceChunker, RecursiveChunker

CHUNK_SIZE = 256
CHUNK_OVERLAP = 32

# ====================================================
# Chunking configuration object
# > Defines: name, chunker instance, and output file
# ====================================================
@dataclass
class ChunkingConfig:
    name: str
    chunker: object
    output_file: str
# ==============================
# Available chunking strategies
# Each strategy uses a different splitting logic:
# > Recursive: structure-aware splitting
# > Token: token-based splitting
# > Sentence: sentence-aware splitting
# ==============================
# More info on https://docs.chonkie.ai/oss/chunkers/overview 
CHUNKING_CONFIGS = [
    ChunkingConfig(
        name="recursive",
        chunker=RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            min_characters_per_chunk=64
        ),
        output_file="recursive.json"
    ),
    ChunkingConfig(
        name="token",
        chunker=TokenChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ),
        output_file="token.json"
    ),
    ChunkingConfig(
        name="sentence",
        chunker=SentenceChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        ),
        output_file="sentence.json"
    ),
]

# ==============================
#       I/O UTILITIES
# ==============================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==============================
#       DOCUMENT BUILDING
# ==============================
def build_documents(data):
    """
    Convert raw hierarchical article structure into LlamaIndex Documents.

    This step flattens: articles → paragraphs → subpoints 
    into individual text documents with metadata.
    """
    documents = []

    for article in data:
        article_id = article.get("article_id")
        article_number = article.get("article_number")

        for block in article.get("blocks", []):
            paragraph_id = block.get("id")
            content = block.get("content", {})
            content_type = content.get("type")

            # ------------------------------
            # Case 1: plain text paragraph
            # ------------------------------
            if content_type == "text":
                text = content.get("text", "").strip()

                if text:
                    documents.append(Document(
                        text=text,
                        metadata={
                            "article_id": article_id,
                            "article_number": article_number,
                            "paragraph_id": paragraph_id,
                            "subpoint_ids": [],
                            "citation": f"Art. {article_number}({paragraph_id[1:]})"
                        }
                    ))
            # ----------------------------------
            # Case 2: paragraph with subpoints
            # ----------------------------------
            elif content_type == "subpoints":

                # Add intro as separate document
                intro = content.get("intro", "")
                if intro:
                    documents.append(Document(
                        text=intro,
                        metadata={
                            "article_id": article_id,
                            "article_number": article_number,
                            "paragraph_id": paragraph_id,
                            "subpoint_ids": [],
                            "citation": f"Art. {article_number}({paragraph_id[1:]})"
                        }
                    ))

                # Add each subpoint as separate document
                for item in content.get("items", []):
                    sub_text = item.get("text", "").strip()
                    sub_id = item.get("id")

                    if sub_text:
                        documents.append(Document(
                            text=sub_text,
                            metadata={
                                "article_id": article_id,
                                "article_number": article_number,
                                "paragraph_id": paragraph_id,
                                "subpoint_ids": [sub_id],
                                "citation": f"Art. {article_number}({paragraph_id[1:]})({sub_id})"
                            }
                        ))

    print(f"Loaded {len(documents)} documents")
    return documents

# ==============================
#           CHUNKING 
# ==============================
def chunk_documents(documents, chunker, chunker_name):
    """
    Apply a chunking strategy to each document and produce final chunks.

    Each chunk keeps:
    > Original metadata
    > Chunk-specific ID
    > Chunk index within document
    """

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        # Apply chunker to raw text
        chunks = chunker(doc.text)

        for i, chunk in enumerate(chunks):
            # Copy original metadata from source document
            metadata = doc.metadata.copy()

            # Ensure subpoint structure exists
            metadata["subpoint_ids"] = metadata.get("subpoint_ids") or []

            # Add chunk-level identifiers
            metadata["chunk_id"] = f"{chunker_name}_{chunk_id}"
            metadata["chunk_index"] = i

            article_number = metadata.get("article_number")
            paragraph_id = metadata.get("paragraph_id")
            subpoint_ids = metadata.get("subpoint_ids") or []

            text = chunk.text

            # Create final chunk document
            all_chunks.append(Document(
                text=text,
                metadata=metadata
            ))

            chunk_id += 1

    return all_chunks

# ==============================
#           PIPELINE
# ==============================
def run_chunking_pipeline(input_file: Path, output_dir: Path):
    """
    Full pipeline:
    1. Load raw dataset
    2. Convert to Documents
    3. Apply multiple chunking strategies
    4. Save results per strategy
    """
    # Load structured input data
    raw_data = load_json(input_file)

    # Convert hierarchy → flat documents
    documents = build_documents(raw_data)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all chunking strategies
    for config in CHUNKING_CONFIGS:
        print(f"\nRunning: {config.name}")

        # Apply chunking strategy
        chunks = chunk_documents(documents, config.chunker, config.name)
        output_path = output_dir / config.output_file

        # Prepare JSON output format
        json_data = [
            {
                "chunk": c.text,
                "metadata": c.metadata
            }
            for c in chunks
        ]

        save_json(json_data, output_path)
        print(f"Saved {len(chunks)} chunks → {output_path}")

if __name__ == "__main__":

    BASE = Path(__file__).resolve().parent.parent

    input_file = BASE / "data" / "corpus" / "final_gdpr_articles_hierarchical.json"
    output_dir = BASE / "data" / "chunks" 

    run_chunking_pipeline(input_file, output_dir)