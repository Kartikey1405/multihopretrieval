
def supporting_fact_match(supporting_fact, document):
    """
    Check whether a retrieved document contains a specific
    HotpotQA supporting fact.

    supporting_fact: [title, sentence_id]
    document: LangChain Document
    """
    title, sentence_id = supporting_fact

    retrieved_title = document.metadata.get("title")
    retrieved_sentence_ids = document.metadata.get("sentence_ids", [])

    return (
        title == retrieved_title
        and sentence_id in retrieved_sentence_ids
    )


def calculate_recall_at_k(supporting_facts, retrieved_documents):
    """
    Supporting-fact Recall@K:
    Number of distinct gold supporting facts retrieved
    divided by the total number of distinct gold supporting facts.
    """
    gold_facts = {
        (fact[0], fact[1])
        for fact in supporting_facts
    }

    if not gold_facts:
        return 0.0

    retrieved_facts = set()

    for fact in gold_facts:
        if any(
            supporting_fact_match(fact, document)
            for document in retrieved_documents
        ):
            retrieved_facts.add(fact)

    return len(retrieved_facts) / len(gold_facts)


def calculate_precision_at_k(supporting_facts, retrieved_documents, k=5):
    """
    Chunk-level Precision@K:
    Number of relevant chunks among the top K chunks divided by K.

    A chunk is relevant if it contains at least one gold
    supporting fact. The denominator remains K, even if fewer
    than K documents are returned.
    """
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not supporting_facts:
        return 0.0

    top_k_documents = retrieved_documents[:k]

    relevant_count = sum(
        1
        for document in top_k_documents
        if any(
            supporting_fact_match(fact, document)
            for fact in supporting_facts
        )
    )

    return relevant_count / k


def calculate_mrr(supporting_facts, retrieved_documents):
    """
    Supporting-fact MRR:
    For each gold supporting fact, find the rank of the first
    retrieved chunk containing it. Average reciprocal ranks
    across the gold supporting facts.

    Note: This is supporting-fact-level MRR, not conventional
    query-level MRR.
    """
    gold_facts = {
        (fact[0], fact[1])
        for fact in supporting_facts
    }

    if not gold_facts:
        return 0.0

    reciprocal_ranks = []

    for fact in gold_facts:
        for rank, document in enumerate(retrieved_documents, start=1):
            if supporting_fact_match(fact, document):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
