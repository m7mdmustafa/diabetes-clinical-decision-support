import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import search_guidelines


# ==========================================
# 1. Load Gemini API Key
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found")

client = genai.Client(api_key=api_key)


# ==========================================
# 2. Retrieval Settings
# ==========================================

TOP_K = 3


# ==========================================
# 3. System Prompt
# ==========================================

SYSTEM_PROMPT = """
You are an evidence-grounded clinical decision support assistant.

Your job is to answer questions using ONLY the retrieved guideline evidence.

Rules:

1. Use only the evidence provided in the prompt.
2. Do not use outside medical knowledge.
3. Do not invent recommendations.
4. Do not provide patient-specific diagnosis or treatment.
5. Every clinical recommendation or factual claim must have a citation.
6. Use only the SOURCE labels provided in the evidence.
7. Do not invent SOURCE numbers.
8. If the evidence is insufficient, say exactly:
   "The retrieved evidence is insufficient to answer this question."
9. Support clinicians and do not replace clinical judgment.
10. Keep the answer concise and easy to inspect.
"""


# ==========================================
# 4. Build Evidence Context
# ==========================================

def build_context(results):

    context_parts = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for i in range(len(documents)):

        metadata = metadatas[i]

        source = f"""
SOURCE {i + 1}

Chunk ID: {ids[i]}
Document: {metadata['document']}
Section: {metadata['section']}
Page: {metadata['page']}
Retrieval distance: {distances[i]:.4f}

Evidence:
{documents[i]}
"""

        context_parts.append(source)

    return "\n".join(context_parts)


# ==========================================
# 5. Build Source List
# ==========================================

def build_source_list(results):

    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    distances = results["distances"][0]

    sources = []

    for i in range(len(metadatas)):

        metadata = metadatas[i]

        source = (
            f"[SOURCE {i + 1}] "
            f"Chunk: {ids[i]} | "
            f"Document: {metadata['document']} | "
            f"Section: {metadata['section']} | "
            f"Page: {metadata['page']} | "
            f"Distance: {distances[i]:.4f}"
        )

        sources.append(source)

    return "\n".join(sources)


# ==========================================
# 6. Generate Grounded Answer
# ==========================================

def generate_answer(question):

    # Step 1: Retrieve Top-3 evidence
    results = search_guidelines(
        question,
        top_k=TOP_K
    )

    # Step 2: Build evidence context
    context = build_context(results)

    # Step 3: Build grounding prompt
    prompt = f"""
Clinical Question:
{question}

Retrieved Guideline Evidence:
{context}

Answer the clinical question using ONLY the retrieved evidence.

IMPORTANT CITATION RULES:

1. Every clinical recommendation or factual claim must have a citation.
2. Cite the exact SOURCE that supports the claim.
3. Use ONLY these citation labels:
   [SOURCE 1]
   [SOURCE 2]
   [SOURCE 3]
4. Do not invent document names.
5. Do not invent page numbers.
6. Do not invent SOURCE labels.
7. If a claim is not supported by the retrieved evidence, do not include it.
8. If the retrieved evidence is insufficient, say:
   "The retrieved evidence is insufficient to answer this question."

Use this format:

Answer:
- Claim or recommendation [SOURCE X]
- Claim or recommendation [SOURCE X]

Do not create a separate Sources section.
The Python program will attach the verified source metadata.
"""

    # Step 4: Ask Gemini
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=2000,
            temperature=0.2,
        )
    )

    # Debug information
    print("\nDEBUG RESPONSE:")
    print("Finish reason:", response.candidates[0].finish_reason)

    answer = response.text

    # Step 5: Build verified sources
    sources = build_source_list(results)

    return answer, sources


# ==========================================
# 7. Run Program
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nAsk a Type 2 Diabetes question: "
    )

    answer, sources = generate_answer(question)

    print("\n")
    print("=" * 60)
    print("GROUNDED ANSWER")
    print("=" * 60)

    print(answer)

    print("\n")
    print("=" * 60)
    print("VERIFIED SOURCES")
    print("=" * 60)

    print(sources)
