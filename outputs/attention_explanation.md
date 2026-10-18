# Self-Attention Engine — Explanation

## Overview
The mRNA-VIP attention module demonstrates scaled dot-product self-attention on an RNA nucleotide sequence. It accepts sequences containing A, C, G, and U, with lengths between 5 and 50 nucleotides.

## Computation
Each nucleotide is converted into a four-dimensional one-hot representation. Query (Q), key (K), and value (V) vectors are computed using projection matrices. The attention scores are calculated as QKᵀ / √dₖ, and a numerically stable row-wise softmax converts them into attention weights. The output is the weighted sum of the value vectors.

## Validation
The module checks sequence characters and length, verifies that the attention matrix has shape n × n, checks that weights are finite and non-negative, and confirms that every row sums to approximately 1.

## Limitations
The projection matrices are randomly initialized and are not trained on biological data. Consequently, the resulting attention weights demonstrate the mathematical mechanism only. They must not be interpreted as learned biological interactions, predictors of vaccine safety, or clinical evidence.

## Output
The heatmap is saved at `outputs/figures/attention_heatmap.png`.
