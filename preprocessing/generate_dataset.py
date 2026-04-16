import json
from pathlib import Path
from typing import List, Dict, Any

counter = 1

def next_id():
    global counter
    qid = counter
    counter += 1
    return qid
    
# =========================
#       IO UTILITIES
# =========================
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# =========================
#   PARAGRAPH PROCESSING
# =========================
def process_paragraph(article_number: str, paragraph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts a single paragraph into one QA pair.

    Two supported formats:
    1. text paragraph → direct QA
    2. subpoints paragraph → structured QA with conditions
    """
    results = []

    paragraph_id = paragraph["id"]
    content = paragraph["content"]

    # CASE 1: Simple text paragraph
    if content["type"] == "text":
        results.append({
            "question_id": next_id(),
            "question": f"What does Article {article_number}, paragraph {paragraph_id} regulate?",
            "answer": content["text"],
            "evidence": {
                "article_number": article_number,
                "paragraph_id": paragraph_id,
                "subpoint_ids": []
            }
        })
        return results

    # CASE 2: Structured subpoints paragraph 
    if content["type"] == "subpoints":
        subpoints = content["items"]
        subpoint_texts = []
        subpoint_ids = []

        for sp in subpoints:
            label = f"({sp['id']})"
            text = f"{label} {sp['text']}"
            
            subpoint_texts.append(sp["text"])
            subpoint_ids.append(sp["id"])

        results.append({
            "question_id": next_id(),
            "question": f"What conditions are defined in Article {article_number}, paragraph {paragraph_id}?",
            "answer": {
                "paragraph_intro": content.get("intro", ""),
                "conditions": subpoint_texts
            },
            "evidence": {
                "article_number": article_number,
                "paragraph_id": paragraph_id,
                "subpoint_ids": subpoint_ids
            }
        })

        return results

    return results

# =========================
#   ARTICLE PROCESSING
# =========================
def process_article(article: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Processes all paragraphs in an article and generates QA pairs.
    """
    results = []

    article_number = article["article_number"]

    for paragraph in article["blocks"]:
        results.extend(process_paragraph(article_number, paragraph))

    return results

# ====================
#   COVERAGE PHASE
# ====================
def generate_coverage_qas(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generates 1 QA per paragraph across all articles.

    This ensures:
    - full dataset coverage
    - deterministic QA generation
    """
    results = []

    for article in articles:
        for paragraph in article["blocks"]:
            para_qas = process_paragraph(article["article_number"], paragraph)
            if para_qas:
                results.append(para_qas[0])
    return results

# =========================
#     ENSURE COVERAGE
# =========================
def ensure_all_articles_covered(results: List[Dict[str, Any]], articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    covered = set()
    for qa in results:
        if "evidence" in qa and "article_number" in qa["evidence"]:
            covered.add(qa["evidence"]["article_number"])

    for article in articles:
        if article["article_number"] not in covered:
            article_number = article["article_number"]

            # force at least 1 QA
            para_qas = process_paragraph(article_number, article["blocks"][0])
            if para_qas:
                results.append(para_qas[0])

    return results

# =========================
#       MAIN PIPELINE
# =========================
def run_pipeline(input_path: Path, output_path: Path):
    articles = load_json(input_path)
    results = generate_coverage_qas(articles)
    results = ensure_all_articles_covered(results, articles)

    save_json(results, output_path)
    print("Saved to:", output_path)

if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent.parent

    input_file = BASE / "data" / "corpus" / "final_gdpr_articles_hierarchical.json"
    output_file = BASE / "data" / "queries" / "gdpr_qa_dataset.json"

    run_pipeline(input_file, output_file)