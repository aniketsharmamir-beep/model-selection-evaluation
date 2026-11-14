# Model Selection and Evaluation Report

## Objective
Compare three machine learning models on the engineered Titanic dataset and choose the best one for predicting passenger survival.

## Method
- Data: engineered_features.csv (891 passengers, 18 features)
- Split: 80% training, 20% testing (random_state=42, stratified)
- Models: Logistic Regression, Decision Tree, Random Forest
- Metrics: accuracy, precision, recall, F1-score
- Validation: 5-fold cross-validation on the training data

## Results
| Model | Train Acc | CV Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.824 | 0.813 | 0.844 | 0.825 | 0.754 | 0.788 |
| Decision Tree | 0.983 | 0.791 | 0.799 | 0.770 | 0.681 | 0.723 |
| Random Forest | 0.983 | 0.812 | 0.816 | 0.773 | 0.739 | 0.756 |

The comparison chart is in images/model_comparison.png.

## Analysis
- The Decision Tree overfits: it scores 0.983 on the training data but only 0.791 in cross-validation, which means it memorized the training passengers instead of learning general patterns.
- The Random Forest also fits the training data almost perfectly (0.983), but it holds up much better on unseen data (CV 0.812) because averaging 100 trees reduces overfitting.
- Logistic Regression has similar training and validation scores (0.824 and 0.813), so it does not overfit.

## Conclusion
Logistic Regression is the chosen model. It has the best cross-validated accuracy (0.813, narrowly ahead of the Random Forest at 0.812), the best test accuracy and F1-score, and it is the simplest and easiest to explain. The Random Forest is a close second and is slightly more stable across folds. Because the test set is small (179 passengers), differences of one or two percent are within normal noise, so the simpler model is the safer pick.
