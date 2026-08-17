from retrieve import search_guidelines


QUESTIONS = [
    "What are the treatment options for type 2 diabetes?",
    "When should medication be started for type 2 diabetes?",
    "What lifestyle changes are recommended for type 2 diabetes?",
    "How should blood glucose be monitored in type 2 diabetes?",
    "What factors should be considered when choosing treatment?"
]


# Our manual relevance judgments
# True = relevant
# False = not relevant
RELEVANCE = {
    1: [True, True, True, False, True],
    2: [True, True, True, False, True],
    3: [True, True, False, False, False],
    4: [True, True, True, False, False],
    5: [True, False, False, True, True],
}


def precision_at_k(relevant_results, k):
    relevant_count = sum(relevant_results[:k])
    return relevant_count / k


def run_evaluation():

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    for question_number, question in enumerate(QUESTIONS, start=1):

        print(f"\nQuestion {question_number}:")
        print(question)

        results = search_guidelines(
            question,
            top_k=10
        )

        relevance = RELEVANCE[question_number]

        p3 = precision_at_k(relevance, 3)
        p5 = precision_at_k(relevance, 5)

        print(f"Precision@3: {p3:.2f}")
        print(f"Precision@5: {p5:.2f}")


if __name__ == "__main__":
    run_evaluation()
