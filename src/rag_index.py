"""Local semantic retrieval for the mRNA-VIP educational reference."""

from pathlib import Path
import re
import time

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDELINES_PATH = PROJECT_ROOT / "data" / "mrna_safety_guidelines.txt"
DB_PATH = PROJECT_ROOT / "outputs" / "chroma_db"
COLLECTION_NAME = "mrna_safety_guidelines"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

_model = None
_client = None
_collection = None


def load_guidelines(path: Path = GUIDELINES_PATH) -> str:
    """Load the educational safety reference text."""
    if not path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Guidelines file is empty.")
    return text


def split_into_paragraphs(text: str) -> list[str]:
    """Split source text into non-empty paragraphs."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def chunk_document(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create section-aware chunks from the source paragraphs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_section = "General"
    current_words = []
    current_start = 0
    total_words = 0

    def save_chunk(words, start_word, end_word, section):
        if not words:
            return

        chunks.append({
            "chunk_id": len(chunks),
            "text": " ".join(words),
            "source": GUIDELINES_PATH.name,
            "section": section,
            "start_word": start_word,
            "end_word": end_word,
        })

    for paragraph in paragraphs:
        heading_match = re.match(r"Section\s+\d+:", paragraph, re.IGNORECASE)

        if heading_match:
            save_chunk(
                current_words, current_start, total_words, current_section
            )
            current_words = []
            current_section = paragraph.strip()
            current_start = total_words

        paragraph_words = paragraph.split()

        if not paragraph_words:
            continue

        if len(paragraph_words) > chunk_size:
            save_chunk(
                current_words, current_start, total_words, current_section
            )
            current_words = []
            position = 0

            while position < len(paragraph_words):
                piece = paragraph_words[position:position + chunk_size]
                save_chunk(
                    piece,
                    total_words + position,
                    total_words + position + len(piece),
                    current_section,
                )
                if position + chunk_size >= len(paragraph_words):
                    break
                position += chunk_size - overlap

            total_words += len(paragraph_words)
            current_start = total_words
            continue

        if current_words and len(current_words) + len(paragraph_words) > chunk_size:
            save_chunk(
                current_words, current_start, total_words, current_section
            )
            current_words = current_words[-overlap:] if overlap else []
            current_start = total_words - len(current_words)

        if not current_words:
            current_start = total_words

        current_words.extend(paragraph_words)
        total_words += len(paragraph_words)

    save_chunk(current_words, current_start, total_words, current_section)

    return chunks
def get_embedding_model():
    """Load the embedding model once per Python process."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_vector_database(reset: bool = False):
    """Build or reuse a persistent local ChromaDB collection."""
    global _client, _collection

    chunks = chunk_document(
        load_guidelines(),
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    DB_PATH.mkdir(parents=True, exist_ok=True)

    if reset:
        _client = None
        _collection = None

    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_PATH))

    if reset:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Rebuild if the index is empty or its chunk count no longer matches.
    if _collection.count() != len(chunks):
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        _collection = _client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        model = get_embedding_model()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

        _collection.add(
            ids=[f"chunk-{chunk['chunk_id']}" for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                }
                for chunk in chunks
            ],
        )

    return _client, _collection


def retrieve(query: str, top_k: int = 3) -> dict:
    """Retrieve relevant chunks and measure query latency."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Exclude first-time model/database initialization from query latency.
    model = get_embedding_model()
    _, collection = build_vector_database()

    start_time = time.perf_counter()
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    matches = []
    for i, document in enumerate(results["documents"][0]):
        matches.append(
            {
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return {
        "query": query,
        "matches": matches,
        "latency_ms": round(latency_ms, 2),
    }


if __name__ == "__main__":
    _, collection = build_vector_database(reset=True)
    print(f"Indexed chunks: {collection.count()}")
    result = retrieve("What are the storage and handling requirements?", top_k=3)
    print(f"Query latency: {result['latency_ms']} ms")
    for rank, match in enumerate(result["matches"], start=1):
        print(f"\nRank {rank} | {match['metadata']['section']}")
        print(f"Distance: {match['distance']:.4f}")
        print(match["text"][:300])
