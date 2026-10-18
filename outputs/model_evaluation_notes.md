# Model Evaluation — mRNA-VIP

## Experimental setup
- Dataset: 500 records
- Target: Reaction_Score (binary classification)
- Train/test split: 80% / 20%
- Stratification: enabled
- Random seed: 42
- Models: Logistic Regression and Random Forest

## Results

| Model | F1-score | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.701 | 0.791 |
| Random Forest | 0.645 | 0.759 |

## Interpretation
Logistic Regression achieved the higher F1-score and ROC-AUC on this held-out test split, so it was selected by the current training script.

These results describe one split of a small educational dataset. They do not establish generalization to other populations or real-world vaccine safety outcomes. Cross-validation and external validation would be needed for stronger performance claims.

## Reproducibility
Run `python src/risk_model.py` from the project root to regenerate the model comparison, confusion matrices, ROC curve, and saved model.

Educational prototype only; not for clinical decision-making.
