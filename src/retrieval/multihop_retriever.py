
import re

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker


class MultiHopRetriever:
    """
    Iterative multi-hop retriever.

    Stage 1: Retrieve documents using the original question.
    Stage 2: Retrieve documents for comparison entities and
             expand using titles from the initial results.
    Final: Merge candidates and rerank with the cross-encoder.
    """

    def __init__(
        self,
        candidate_k=20,
        initial_k=10,
        expansion_k=5,
        final_k=15,
        rrf_k=60,
    ):
        self.candidate_k = candidate_k
        self.initial_k = initial_k
        self.expansion_k = expansion_k
        self.final_k = final_k

        self.hybrid_retriever = HybridRetriever(rrf_k=rrf_k)
        self.reranker = Reranker()

    def _extract_comparison_titles(self, query):
        """Extract two likely entities from simple comparison questions."""

        patterns = [
            r"\bof\s+(.+?)\s+and\s+(.+?)\s+(?:the same|same)\b",
            r"\bbetween\s+(.+?)\s+and\s+(.+?)[?.!]*$",
            r"\bwere\s+(.+?)\s+and\s+(.+?)\s+the same\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                first = match.group(1).strip(" ?.")
                second = match.group(2).strip(" ?.")
                if first and second:
                    return [first, second]

        return []

    def _document_key(self, document):
        return (
            document.metadata.get("title", ""),
            document.metadata.get("source_id", ""),
            document.page_content,
        )

    def _get_expansion_titles(self, documents):
        titles = []
        seen = set()

        for document in documents:
            title = document.metadata.get("title", "").strip()

            if not title:
                continue

            normalized_title = title.lower()

            if normalized_title in seen:
                continue

            seen.add(normalized_title)
            titles.append(title)

            if len(titles) >= self.expansion_k:
                break

        return titles

    def retrieve(self, query, k=None):
        """Retrieve, expand, deduplicate, and rerank evidence."""

        if k is None:
            k = self.final_k

        # Stage 1: retrieve for the original question.
        initial_documents = self.hybrid_retriever.retrieve(
            query,
            k=self.initial_k,
            candidate_k=self.candidate_k,
        )

        # Keep all candidates in a dictionary for deduplication.
        candidate_documents = {}

        for document in initial_documents:
            candidate_documents[self._document_key(document)] = document

        # Stage 2A: directly retrieve comparison entities.
        comparison_titles = self._extract_comparison_titles(query)

        for title in comparison_titles:
            expanded_documents = self.hybrid_retriever.retrieve(
                title,
                k=self.candidate_k,
                candidate_k=self.candidate_k,
            )

            for document in expanded_documents:
                candidate_documents[self._document_key(document)] = document

        # Stage 2B: expand using titles from initial retrieval.
        expansion_titles = self._get_expansion_titles(initial_documents)

        for title in expansion_titles:
            expanded_query = f"{query} {title}"

            expanded_documents = self.hybrid_retriever.retrieve(
                expanded_query,
                k=self.candidate_k,
                candidate_k=self.candidate_k,
            )

            for document in expanded_documents:
                candidate_documents[self._document_key(document)] = document

        candidates = list(candidate_documents.values())

        if not candidates:
            return []

        # Final stage: rerank against the original question.
        return self.reranker.rerank(
            query,
            candidates,
            k=k,
        )
