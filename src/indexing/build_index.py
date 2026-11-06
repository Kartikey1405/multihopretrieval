import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.indexing.embeddings import get_embedding_model

CHUNKS_PATH = "data/processed/chunks.json"
INDEX_PATH = "vectorstore/faiss"
BATCH_SIZE = 64


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index():
    print("Loading chunks...", flush=True)
    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks.", flush=True)

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "title": chunk["title"],
                "source_id": chunk["source_id"],
                "sentence_ids": chunk["sentence_ids"],
            },
        )
        for chunk in chunks
    ]

    print(f"Created {len(documents):,} documents.", flush=True)
    print("Loading embedding model...", flush=True)
    embeddings = get_embedding_model()

    vectorstore = None
    total = len(documents)

    for start in range(0, total, BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        print(
            f"Embedding documents {start + 1:,}–"
            f"{start + len(batch):,} of {total:,}...",
            flush=True,
        )

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    print("Saving FAISS index...", flush=True)
    Path(INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"FAISS index saved to: {INDEX_PATH}", flush=True)


if __name__ == "__main__":
    build_index()