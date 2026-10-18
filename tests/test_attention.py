

import numpy as np
import pytest

from src.attention import compute_self_attention, validate_sequence


def test_attention_weights_sum_to_one():
    result = compute_self_attention("ACGUAC")
    weights = result["attention_weights"]

    assert weights.shape == (6, 6)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_attention_weights_are_nonnegative():
    result = compute_self_attention("ACGUAC")
    assert np.all(result["attention_weights"] >= 0)


def test_attention_output_shape():
    result = compute_self_attention("ACGUAC", embedding_dim=8)

    assert result["attention_output"].shape == (6, 8)
    assert result["queries"].shape == (6, 8)
    assert result["keys"].shape == (6, 8)
    assert result["values"].shape == (6, 8)


def test_attention_is_reproducible():
    first = compute_self_attention("ACGUAC", seed=42)
    second = compute_self_attention("ACGUAC", seed=42)

    np.testing.assert_allclose(
        first["attention_weights"],
        second["attention_weights"],
    )


def test_sequence_is_normalized():
    assert validate_sequence("  acguac  ") == "ACGUAC"


@pytest.mark.parametrize("sequence", ["", "ACGU", "ACGU" * 13])
def test_sequence_length_validation(sequence):
    with pytest.raises(ValueError):
        validate_sequence(sequence)


@pytest.mark.parametrize("sequence", ["ACGTAC", "ACGNAC"])
def test_invalid_nucleotides_are_rejected(sequence):
    with pytest.raises(ValueError):
        validate_sequence(sequence)


def test_non_string_sequence_is_rejected():
    with pytest.raises(TypeError):
        validate_sequence(None)


def test_invalid_embedding_dimension_is_rejected():
    with pytest.raises(ValueError):
        compute_self_attention("ACGUAC", embedding_dim=0)
