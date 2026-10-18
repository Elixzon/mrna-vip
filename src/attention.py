
"""NumPy self-attention engine for the mRNA-VIP educational project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

VALID_NUCLEOTIDES = set("ACGU")
MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 50


def validate_sequence(sequence: str) -> str:
    """Validate and normalize an RNA nucleotide sequence."""
    if not isinstance(sequence, str):
        raise TypeError("Sequence must be a string.")

    sequence = sequence.strip().upper()

    if not MIN_SEQUENCE_LENGTH <= len(sequence) <= MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length must be between "
            f"{MIN_SEQUENCE_LENGTH} and {MAX_SEQUENCE_LENGTH}."
        )

    invalid_characters = sorted(set(sequence) - VALID_NUCLEOTIDES)
    if invalid_characters:
        raise ValueError(
            f"Invalid nucleotide characters: {invalid_characters}. "
            "Only A, C, G, and U are allowed."
        )

    return sequence

def compute_self_attention(
    sequence: str,
    embedding_dim: int = 8,
    seed: int = 42,
) -> dict:
    """Compute scaled dot-product self-attention for an RNA sequence."""
    sequence = validate_sequence(sequence)

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1.")

    rng = np.random.default_rng(seed)
    sequence_length = len(sequence)

    # One-hot encode each nucleotide: A, C, G, U.
    nucleotide_to_index = {"A": 0, "C": 1, "G": 2, "U": 3}
    encoded = np.eye(4, dtype=float)[
        [nucleotide_to_index[nucleotide] for nucleotide in sequence]
    ]

    # Initialize reproducible projection matrices.
    scale = 1.0 / np.sqrt(4)
    W_q = rng.normal(0, scale, size=(4, embedding_dim))
    W_k = rng.normal(0, scale, size=(4, embedding_dim))
    W_v = rng.normal(0, scale, size=(4, embedding_dim))

    # Project nucleotide representations into queries, keys, and values.
    queries = encoded @ W_q
    keys = encoded @ W_k
    values = encoded @ W_v

    # Scaled dot-product attention.
    scores = queries @ keys.T / np.sqrt(embedding_dim)

    # Numerically stable row-wise softmax.
    shifted_scores = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted_scores)
    attention_weights = exponentials / exponentials.sum(
        axis=1, keepdims=True
    )

    # Combine values according to the attention weights.
    attention_output = attention_weights @ values

    return {
        "sequence": sequence,
        "queries": queries,
        "keys": keys,
        "values": values,
        "attention_weights": attention_weights,
        "attention_output": attention_output,
        "sequence_length": sequence_length,
        "embedding_dim": embedding_dim,
    }

def save_attention_heatmap(
    result: dict,
    output_path: Path | None = None,
) -> Path:
    """Save an attention-weight heatmap for the supplied sequence."""
    if output_path is None:
        output_path = FIGURE_DIR / "attention_heatmap.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequence = result["sequence"]
    weights = result["attention_weights"]

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        weights,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )

    positions = np.arange(len(sequence))
    ax.set_xticks(positions, labels=list(sequence))
    ax.set_yticks(positions, labels=list(sequence))
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("RNA Self-Attention Weights")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Attention weight")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path

def run_self_tests() -> None:
    """Verify attention dimensions, normalization, and input validation."""
    test_sequence = "ACGUACGU"
    result = compute_self_attention(test_sequence)

    weights = result["attention_weights"]
    length = len(test_sequence)

    assert weights.shape == (length, length), (
        f"Unexpected attention shape: {weights.shape}"
    )
    assert np.all(np.isfinite(weights)), "Attention weights must be finite."
    assert np.all(weights >= 0), "Attention weights must be non-negative."
    assert np.allclose(weights.sum(axis=1), 1.0), (
        "Each attention row must sum to 1."
    )
    assert result["attention_output"].shape == (
        length,
        result["embedding_dim"],
    )

    for invalid_sequence in ("ACGTAC", "ACGXAC", "ACG", "A" * 51):
        try:
            validate_sequence(invalid_sequence)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid sequence was accepted: {invalid_sequence}"
            )

    print("All self-attention tests passed.")
    print(f"Sequence length: {length}")
    print(f"Attention matrix shape: {weights.shape}")
    print(f"Attention row sums: {weights.sum(axis=1).round(6)}")


if __name__ == "__main__":
    run_self_tests()
    test_result = compute_self_attention("ACGUACGU")
    heatmap_path = save_attention_heatmap(test_result)
    print(f"Saved attention heatmap: {heatmap_path}")
