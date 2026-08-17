import chromadb
from sentence_transformers import SentenceTransformer

from ingest import extract_and_chunk_pdfs, PDF_FILES


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create local Chroma database
client = chromadb.PersistentClient(path="chroma_db")


# Create or load collection
collection = client.get_or_create_collection(
    name="diabetes_guidelines"
)


def build_vector_database():

    print("Reading PDFs and creating chunks...")

    chunks = extract_and_chunk_pdfs(PDF_FILES)

    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    ).tolist()

    ids = [
        chunk["chunkID"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "document": chunk["document"],
            "section": chunk["section"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Vector database created successfully!")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_database()
