import json
import os
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data/queries/gdpr_qa_dataset.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/queries/rewritten_queries.json")

# ============================================================
# Dataset loader: handles normal JSON + deouble encoded JSON
# ============================================================
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle case where JSON is encoded as a string inside JSON
    if isinstance(data, str):
        data = json.loads(data)

    return data

# =============================================================
# Extract structured fields from dataset item -
# normalizes different possible formats "evidence" and "answer"
# =============================================================
def extract_fields(item):
    evidence = item.get("evidence", {})

    # Extract article number
    if isinstance(evidence, dict):
        article = evidence.get("article_number", "")
    else:
        article = ""

    answer = item.get("answer", "")

    # Handle structured answer format
    if isinstance(answer, dict):
        paragraph_text = answer.get("paragraph_intro", "")
        subpoints = answer.get("conditions", [])
    # Handle string answer format
    elif isinstance(answer, str):
        paragraph_text = answer
        subpoints = []
    # Default for unexpected formats
    else:
        paragraph_text = ""
        subpoints = []

    return article, paragraph_text, subpoints

# ================================================================
# Prompt builder 
# > Converts QA + legal structure into retrieval-friendly-
# instruction (natural language)
# > Enforces strict rewriting rules for consistency
# ================================================================
def build_prompt(question, article, paragraph_text, subpoints):
    return f"""
You are rewriting legal QA questions into a single clear, standalone search query.

STRICT RULES:

- Always include "Article {article}"
- Use natural, fluent language (not keyword lists)

CASE 1 — NORMAL PARAGRAPH (no subpoints):
- Extract 1–2 keywords PER sentence from the paragraph text

CASE 2 — INTRO PARAGRAPH (with subpoints):
- Extract 1–2 keywords from EACH paragraph_intro
- Extract 1-2 keywords from EACH subpoint
- ALL subpoint keywords must be included in the final question

GENERAL RULES:
- A keyword is 1–2 words representing a topic/subject
- Combine all keywords into one natural question

- Do NOT copy full sentences verbatim
- Do NOT include IDs (p1, sa, etc.)
- Do NOT use vague pronouns ("this regulation", "it", "the regulation")
- Do NOT create follow-up or conversational questions
- Each question must be fully self-contained
- Preserve original meaning exactly
- Use minimal wording for each keyword

QUESTION:
{question}

PARAGRAPH:
{paragraph_text}

SUBPOINTS:
{subpoints}

Return ONLY the rewritten question.
"""

# ==============================
# OpenAI API call
# > Sends prompt to GPT model
# > Returns rewritten query
# ==============================
def rewrite_question(item):
    question = item.get("question", "")

    # Extract structured components
    article, paragraph_text, subpoints = extract_fields(item)

    # Build prompt for LLM
    prompt = build_prompt(question, article, paragraph_text, subpoints)

    # Call OpenAI model (gpt4)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You rewrite legal QA into clean retrieval-ready queries."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Return cleaned output
    return response.choices[0].message.content.strip()


# ==============================
#       Main pipeline
# ==============================
def process_dataset():
    data = load_data(INPUT_PATH)
    results = []

    # Iterate over dataset with progress bar (visual)
    for item in tqdm(data):
        # Skip invalid entries (prevents crash)
        if not isinstance(item, dict):
            continue

        try:
            # Rewrite question using LLM
            rewritten = rewrite_question(item)

            # Copy original item and replace question
            new_item = item.copy()
            new_item["question"] = rewritten
            results.append(new_item)

        except Exception as e:
            # Prevent pipeline crash on single failure
            print(f"Skipping item {item.get('question_id', 'unknown')} due to error: {e}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save rewritten dataset
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved rewritten dataset to:\n{OUTPUT_PATH}")

if __name__ == "__main__":
    process_dataset()