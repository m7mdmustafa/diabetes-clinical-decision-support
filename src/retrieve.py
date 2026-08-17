import chromadb
from sentence_transformers import SentenceTransformer


# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to our existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="diabetes_guidelines"
)


TOP_K = 3


def search_guidelines(question, top_k=TOP_K):

    # Convert the user's question into an embedding
    question_embedding = model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    question = input(
        "\nAsk a Type 2 Diabetes question: "
    )

    results = search_guidelines(question)

    print("\n" + "=" * 60)
    print("RETRIEVED EVIDENCE")
    print("=" * 60)

    for i in range(len(results["documents"][0])):

        print(f"\n--- Result {i + 1} ---")

        print(
            f"Document: "
            f"{results['metadatas'][0][i]['document']}"
        )

        print(
            f"Section: "
            f"{results['metadatas'][0][i]['section']}"
        )

        print(
            f"Page: "
            f"{results['metadatas'][0][i]['page']}"
        )

        print(
            f"Distance: "
            f"{results['distances'][0][i]:.4f}"
        )

        print("\nText:")
        print(
            results["documents"][0][i][:500]
        )
