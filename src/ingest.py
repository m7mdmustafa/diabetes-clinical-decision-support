import re
from pypdf import PdfReader


PDF_FILES = [
    "data/guideline_1.pdf",
    "data/guideline_2.pdf"
]


def extract_and_chunk_pdfs(pdf_files, max_chunk_length=1000):
    all_chunks = []
    seen_text = set()
    chunk_id_counter = 1

    section_pattern = re.compile(
        r"^(?:SECTION\s+\d+|\d+\.\d*\s+[A-Z]|[A-Z\s]{5,})$"
    )

    def save_chunk(text, document, section, page):
        nonlocal chunk_id_counter

        text = text.strip()

        # Ignore very short chunks
        if len(text.split()) < 20:
            return

        # Ignore exact duplicate chunks
        normalized_text = " ".join(text.lower().split())

        if normalized_text in seen_text:
            return

        seen_text.add(normalized_text)

        all_chunks.append({
            "chunkID": f"CHK-{chunk_id_counter:04d}",
            "document": document,
            "section": section,
            "page": page,
            "text": text
        })

        chunk_id_counter += 1

    for pdf_path in pdf_files:

        try:
            reader = PdfReader(pdf_path)
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            continue

        current_section = "Introduction / Prologue"

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""
            lines = text.split("\n")

            current_chunk_text = ""

            for line in lines:

                clean_line = line.strip()

                if not clean_line:
                    continue

                # Detect section headers
                if (
                    section_pattern.match(clean_line)
                    and len(clean_line) < 80
                ):

                    if current_chunk_text:
                        save_chunk(
                            current_chunk_text,
                            pdf_path,
                            current_section,
                            page_number
                        )

                        current_chunk_text = ""

                    current_section = clean_line
                    continue

                current_chunk_text += clean_line + " "

                # Split large chunks
                if len(current_chunk_text) >= max_chunk_length:

                    save_chunk(
                        current_chunk_text,
                        pdf_path,
                        current_section,
                        page_number
                    )

                    current_chunk_text = ""

            # Save remaining text
            if current_chunk_text:

                save_chunk(
                    current_chunk_text,
                    pdf_path,
                    current_section,
                    page_number
                )

    return all_chunks


if __name__ == "__main__":

    chunks = extract_and_chunk_pdfs(PDF_FILES)

    print(f"Total useful chunks created: {len(chunks)}\n")

    for chunk in chunks[:3]:

        print("-" * 40)
        print(f"Chunk ID : {chunk['chunkID']}")
        print(f"Document : {chunk['document']}")
        print(f"Section  : {chunk['section']}")
        print(f"Page     : {chunk['page']}")
        print("-" * 40)
        print(f"Text Preview: {chunk['text'][:150]}...\n")
