
import pytest

from src.rag_index import retrieve, split_into_paragraphs, chunk_document


def test_retrieve_returns_expected_structure():
    result = retrieve("storage and handling requirements", top_k=3)

    assert isinstance(result, dict)
    assert result["query"] == "storage and handling requirements"
    assert "matches" in result
    assert "latency_ms" in result
    assert isinstance(result["matches"], list)
    assert len(result["matches"]) > 0


def test_retrieved_matches_contain_text_and_metadata():
    result = retrieve("adverse reaction reporting", top_k=3)

    for match in result["matches"]:
        assert isinstance(match["text"], str)
        assert match["text"].strip()
        assert isinstance(match["metadata"], dict)
        assert "section" in match["metadata"]
        assert "distance" in match


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="Query"):
        retrieve("   ")


def test_zero_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        retrieve("vaccine safety", top_k=0)


def test_split_into_paragraphs():
    paragraphs = split_into_paragraphs("First paragraph.\n\nSecond paragraph.")

    assert paragraphs == ["First paragraph.", "Second paragraph."]


def test_chunk_document_returns_required_fields():
    chunks = chunk_document(
        "Section 1: Introduction\n\nThis is an educational safety guideline.",
        chunk_size=20,
        overlap=3,
    )

    assert len(chunks) > 0
    assert {"chunk_id", "text", "source", "section",
            "start_word", "end_word"}.issubset(chunks[0])


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document("Some text", chunk_size=0)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        chunk_document("Some text", chunk_size=10, overlap=10)
