import fitz
import re
import json
from pathlib import Path

# ===================================
# STEP 1: Extract raw text from PDF
# ===================================
def extract_text(pdf_path):
    """
    Opens a PDF file and extracts text from all relevant pages.
    In this case, we skip the first 31 pages because the GDPR
    articles start at page 32 (index 31).
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(31, len(doc)): 
        pages.append(doc[page_num].get_text("text"))

    return "\n".join(pages)

# =================================================================
# STEP 2: Clean extracted text (remove noise & formatting issues)
# =================================================================
def clean_text(text):
    """
    Removes headers, footers, page markers, and fixes formatting
    issues caused by PDF extraction.
    """
    # Remove official EU journal headers (date + issue info)
    text = re.sub(
        r"\d{1,2}\.\d{1,2}\.\d{4}\s+L\s+\d+/\d+\s+Official Journal of the European Union\s+EN",
        "",
        text
    )

    # Remove partial/misaligned header fragments
    text = re.sub(
        r"L\s+\d+/\d+\s+Official Journal of the European Union\s+EN",
        "",
        text
    )

    # Remove page references like "L 119/32"
    text = re.sub(r"L\s+\d+/\d+", "", text)

    # Remove specific EU footnote block that interferes with Article 5 parsing
    text = re.sub(
        r"\(1\)\s*Directive\s*\(EU\)\s*2015/1535[^\.]*\.\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove specific Regulation 1049/2001 footnote (Article 79)
    text = re.sub(
        r"\(1\)\s*Regulation\s*\(EC\)\s*No\s*1049/2001[^\.]*\.\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove citation-style references like "17.9.2015, p. 1"
    text = re.sub(
        r"\b\d{1,2}\.\d{4},\s*p\.\s*\d+\)\.?",
        "",
        text
    )

    # Fix broken line breaks inside sentences (join hyphen-free line breaks)
    text = re.sub(r"(?<=\w)\n(?=\w)", " ", text)

    # Normalize all whitespace (multiple spaces/newlines --> single space)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =======================================================================
# STEP 3: Parse structured GDPR elements (chapters, sections, articles)
# =======================================================================
def parse_gdpr(text):
    """
    Parses cleaned GDPR text into structured components:
    - Chapters
    - Sections
    - Articles (1-30 only)

    Each article is linked to its nearest chapter and section.
    """
    results = []

    # Normalize spacing of newlines for consistency
    text = re.sub(r"\n+", "\n", text)
    # -------------------------
    # Identify CHAPTER blocks
    # -------------------------
    chapters = list(re.finditer(
        r"(CHAPTER\s+[IVXLC]+\s+.*?)(?=CHAPTER\s+[IVXLC]+|Section\s+\d+\s+|Article\s+\d+\s+[A-Z]|$)",
        text
    ))

    chapter_spans = [(c.start(), c.group(1)) for c in chapters]

    # -------------------------
    # Identify SECTION blocks
    # -------------------------
    sections = list(re.finditer(
        r"(Section\s+\d+\s+.*?)(?=Section\s+\d+|Article\s+\d+\s+[A-Z]|CHAPTER\s+[IVXLC]+|$)",
        text
    ))

    section_spans = [(s.start(), s.group(1)) for s in sections]

    # --------------------------
    # Identify ARTICLE headers
    # --------------------------
    article_iter = list(re.finditer(
        r"Article\s+(\d+)\s+([A-Z][A-Za-z ,\-–]{5,})",
        text
    ))

    if not article_iter:
        print("No real articles found")
        return []

    # ---------------------------------
    # Build structured article blocks
    # ---------------------------------
    for i, m in enumerate(article_iter):

        # Start position of current article
        start = m.start()

        # Determine end boundary (next article/section/chapter)
        next_positions = []

        # Next article boundary
        if i + 1 < len(article_iter):
            next_positions.append(article_iter[i + 1].start())

        # Next chapter boundary
        for pos, _ in chapter_spans:
            if pos > start:
                next_positions.append(pos)

        # Next section boundary
        for pos, _ in section_spans:
            if pos > start:
                next_positions.append(pos)

        # Final end position
        end = min(next_positions) if next_positions else len(text)

        # Extract full article text block
        article_text = text[start:end].strip()

        # Extract article number (ID)
        article_id = m.group(1)

        # -------------------------------------
        # Find closest chapter before article
        # -------------------------------------
        chapter = None
        for pos, ch in chapter_spans:
            if pos <= start:
                chapter = ch

        # -------------------------------------
        # Find closest section before article
        # -------------------------------------
        section = None
        for pos, sec in section_spans:
            if pos <= start:
                section = sec

        # Store structured result
        results.append({
            "chapter": chapter,
            "section": section,  
            "article_id": article_id,
            "text": article_text
        })

    return results

# ========================================
# STEP 4: Save structured output to JSON
# ========================================
def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ======================
# MAIN EXECUTION FLOW
# ======================
if __name__ == "__main__":
    # -----------------------------
    # Project-root relative paths
    # -----------------------------
    BASE_PATH = Path(__file__).resolve().parent.parent
    RAW_PATH = BASE_PATH / "data" / "pdf"
    OUTPUT_PATH = BASE_PATH / "data" / "corpus"

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    pdf_path = RAW_PATH / "CELEX_32016R0679_EN_TXT.pdf"
    output_path = OUTPUT_PATH / "gdpr_articles.json"

    # Step 1: Extract raw text from pdf
    raw_text = extract_text(pdf_path)

    # Step 2: Clean extracted text
    clean = clean_text(raw_text)

    # Step 3: Parse into structured GDPR components
    structured = parse_gdpr(clean)

    # Step 4: Save output
    if structured:
        save_json(structured, output_path)
        print(f"Saved: {output_path}")
        print(structured[0])
    else:
        print("No data returned")