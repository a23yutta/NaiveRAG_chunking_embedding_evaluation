import re
import json
from pathlib import Path

# ===================================
# REGEX PATTERNS
# Used to detect structure in GDPR article text
# ===================================

PARA_RE = re.compile(r"(?:^|\s)(\d{1,2})\.\s+(?=[A-Z])")
SUB_RE = re.compile(r"\(([a-z])\)\s")

# ===================================
# TEXT CLEANING HELPERS
# Normalize and split extracted text
# ===================================

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def split_by_pattern(text, pattern):
    """
    Splits text into structured chunks based on regex pattern.
    Used for:
    - paragraph splitting (1., 2., 3.)
    - subpoint splitting ((a), (b), (c))
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    chunks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)

        chunks.append({
            "id": m.group(1),
            "text": clean_text(text[start:end])
        })

    return chunks

# ===================================
# CORE PARSER LOGIC
# Converts text into hierarchical structure:
# Article → Paragraph → Subpoints
# ===================================
def parse_subpoints(text):
    """
    Detects whether a paragraph contains structured subpoints.

    Returns:
    - structured subpoints if multiple (a), (b), (c) exist
    - plain text otherwise
    """
    sub_matches = list(SUB_RE.finditer(text))

    if len(sub_matches) > 1:
        first_match = sub_matches[0]

        # Extract intro BEFORE (a)
        intro = clean_text(text[:first_match.start()])

        chunks = split_by_pattern(text, SUB_RE)

        return {
            "type": "subpoints",
            "intro": intro,
            "items": [
                {
                    "type": "subpoint",
                    "id": f"s{c['id']}",
                    "text": c["text"]
                }
                for c in chunks
            ]
        }

    # Free text
    return {
        "type": "text",
        "text": clean_text(text)
    }

def parse_article(article, article_index):
    text = article["text"]

    result = {
        "article_id": article_index, # Internal sequential ID for dataset ordering
        "article_number": article["article_id"], # Official GDPR article reference
        "blocks": []
    }

    para_chunks = split_by_pattern(text, PARA_RE)

    # If no paragraphs → treat as single block
    if not para_chunks:
        parsed = parse_subpoints(text)

        result["blocks"].append({
            "type": "paragraph",
            "id": "p1",
            "content": parsed
        })

        return result

    for p in para_chunks:
        content = parse_subpoints(p["text"])

        result["blocks"].append({
            "type": "paragraph",
            "id": f"p{p['id']}",
            "content": content
        })

    return result

def get_article_number(article):
    return int(article["article_id"])
# ===================================
#           IO UTILITIES
# ===================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ======================
# MAIN EXECUTION FLOW
# ======================
if __name__ == "__main__":
    # -----------------------------
    # Project-root relative paths
    # -----------------------------
    BASE = Path(__file__).resolve().parent.parent
    input_file = BASE / "data" / "corpus" / "gdpr_articles.json"
    output_file = BASE / "data" / "corpus" / "gdpr_articles_hierarchical.json"

    data = load_json(input_file)

    """
    Article filtering rules:

    Included articles:
    - manually selected subset (55 / 99 GDPR articles)
    Exclusions:
    - articles containing inline references like "Article X." 
        where "Article X" appears inside the body text and 
        is followed by a period (sentence-ending dot), 
        which breaks parsing structure
    - articles containing nested subpoint structures not yet supported

    Purpose:
    - ensure clean hierarchical parsing without structural ambiguity
    """
    TARGET_ARTICLES = {"1", "3", "5", "7", "10", "11", "15", "16", "18", "19", "24", "25", "26", "27", "29", "31", "32", "33", "38", "39", "44", "46", "48", "50", "52", "53", "54", "59", "61", "63", "66", "68", "72", "73", "75", "76", "79", "80", "81", "82", "84", "85", "86", "87", "88", "89", "91", "92", "93", "94", "95", "96", "97", "98", "99"}  

    # Sort articles by official GDPR article number (not internal ID)
    filtered_articles = []
    for a in data:
        if a["article_id"] in TARGET_ARTICLES:
            filtered_articles.append(a)
    filtered_articles.sort(key=get_article_number)

    results = []
    for idx, article in enumerate(filtered_articles, start=1):
        parsed = parse_article(article, idx)
        results.append(parsed)
        
    save_json(results, output_file)
    print("Saved to:", output_file)