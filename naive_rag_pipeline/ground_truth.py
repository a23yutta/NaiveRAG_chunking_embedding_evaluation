from typing import List, Dict

# ==============================
#   GROUND TRUTH EXTRACTION
# ==============================
def get_ground_truth_nodes(documents, evidence):
    """
    Retrieve ground-truth documents based on evidence metadata.

    Filters the full document list to only those that match:
    - article_number
    - paragraph_id

    This defines the "relevant set" used for evaluation.
    """
    # Extract target identifiers from evidence
    article = evidence["article_number"]
    paragraph = evidence["paragraph_id"]

    # Filter documents matching the evidence
    return [
        doc for doc in documents
        if doc.metadata["article_number"] == article
        and doc.metadata["paragraph_id"] == paragraph
    ]