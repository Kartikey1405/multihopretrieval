
def build_prompt(question: str, documents) -> str:
    evidence_parts = []

    for i, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "Unknown")
        text = document.page_content.strip()

        evidence_parts.append(
            f"Evidence {i}:\n"
            f"Source: {title}\n"
            f"Content: {text}"
        )

    evidence = "\n\n".join(evidence_parts)

    prompt = f"""You are a precise question-answering assistant for a multi-hop
question-answering system.

Answer the question using only the evidence provided below.

Follow these rules carefully:
1. Identify the exact entities mentioned in the question.
2. Match evidence to the correct entity. Do not confuse similarly named
   movies, people, places, or other entities.
3. For comparison questions, find relevant evidence for BOTH entities
   before reaching a conclusion.
4. Base your answer strictly on the provided evidence. Do not use outside
   knowledge or invent missing facts.
5. Start with a direct answer.
6. For yes/no questions, start with "Yes." or "No." and then give a brief
   explanation of the evidence supporting the conclusion.
7. For other questions, answer directly and include a short explanation
   when useful.
8. If the evidence does not establish the answer, clearly state that the
   available evidence is insufficient. Do not guess.
9. Keep the response natural and concise, usually 1-3 sentences.
10. Do not list every evidence passage or discuss the retrieval process.

Question:
{question}

Evidence:
{evidence}

Answer:"""

    return prompt
