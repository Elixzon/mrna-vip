
"""Local semantic retrieval for the mRNA-VIP educational reference."""

import csv
import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDELINES_PATH = PROJECT_ROOT / "data" / "mrna_safety_guidelines.txt"
DB_PATH = PROJECT_ROOT / "outputs" / "chroma_db_v0_5"
COLLECTION_NAME = "mrna_safety_guidelines"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_model = None
_client = None
_collection = None


def log_query_latency(query: str, latency_ms: float) -> None:
    """Append query latency to a persistent CSV log."""
    log_path = PROJECT_ROOT / "outputs" / "rag_latency_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "question", "latency_ms"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            round(latency_ms, 2),
        ])


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


def get_embedding_model():
    """Load the embedding model once per Python process."""
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
        _model.max_seq_length = 512
        _model.tokenizer.model_max_length = 512

    return _model


def chunk_document(text, chunk_size=400, overlap=50):
    """
    Split guideline text into overlapping token chunks.

    Each chunk is associated with the section containing its first token.
    Short sections can produce chunks below the target size; source text
    is never padded or invented to meet a minimum.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    if not text or not text.strip():
        return []

    model = get_embedding_model()
    tokenizer = model.tokenizer

    # Split at section headings to keep chunk content within its section.
    section_pattern = r"(?=Section \d+:)"
    parts = re.split(section_pattern, text)

    chunks = []
    step = chunk_size - overlap

    for part in parts:
        part = part.strip()
        if not part:
            continue

        first_line = part.splitlines()[0].strip()
        if re.match(r"Section \d+:", first_line):
            section = first_line
        else:
            section = "Introduction"

        token_ids = tokenizer.encode(
            part,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )

        position = 0

        while position < len(token_ids):
            piece = token_ids[position:position + chunk_size]
            if not piece:
                break

            chunk_text = tokenizer.decode(
                piece,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": chunk_text,
                    "source": GUIDELINES_PATH.name,
                    "section": section,
                    "start_word": position,
                    "end_word": position + len(piece),
                })

            if position + chunk_size >= len(token_ids):
                break

            position += step

    return chunks


def build_vector_database(reset: bool = False):
    """Build or reuse a persistent local ChromaDB collection."""
    global _client, _collection

    source_text = load_guidelines()
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    chunks = chunk_document(
        source_text,
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
        metadata={"hnsw:space": "cosine", "source_sha256": source_hash},
    )

    # Rebuild when the source document or its chunk count has changed.
    if (
        _collection.count() != len(chunks)
        or _collection.metadata.get("source_sha256") != source_hash
    ):
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        _collection = _client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "source_sha256": source_hash},
        )

        if chunks:
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

    if collection.count() == 0:
        log_query_latency(query, 0.0)
        return {"query": query, "matches": [], "latency_ms": 0.0}

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
    log_query_latency(query, latency_ms)

    matches = []
    for i, document in enumerate(results["documents"][0]):
        matches.append({
            "text": document,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return {
        "query": query,
        "matches": matches,
        "latency_ms": round(latency_ms, 2),
    }


def query_guidelines(question: str, k: int = 3) -> list[dict]:
    """Return the top-k relevant guideline chunks for a question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    result = retrieve(question, top_k=k)
    return result["matches"]


if __name__ == "__main__":
    _, collection = build_vector_database(reset=True)
    print(f"Indexed chunks: {collection.count()}")

    result = retrieve(
        "What are the storage and handling requirements?",
        top_k=3,
    )
    print(f"Query latency: {result['latency_ms']} ms")

    for rank, match in enumerate(result["matches"], start=1):
        print(f"\nRank {rank} | {match['metadata']['section']}")
        print(f"Distance: {match['distance']:.4f}")
        print(match["text"][:300])
