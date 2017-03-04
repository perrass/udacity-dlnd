## Model Evaluation and Validation

### Confusion Matrix

|             | Diagnosed Sick | Diagnosed Healty |
| :---------: | :------------: | :--------------: |
|  **SICK**   | True Positive  |  False Negative  |
| **HEALTHY** | False Positive |  True Negative   |

* True/False means whether the prediction is correct
* Positive/Negative means whether the prediction is 1 (sick) or 0 (healthy)
* False Negative means the prediction is Healthy but the fact is Sick
* False Positive means the prediction is Sick but the fact is Healthy

---

|          | Guessed Positive | Guessed Negative |
| -------- | ---------------- | ---------------- |
| POSITIVE | True Positives   | False Negatives  |
| NEGATIVE | False Positives  | True Negatives   |

* Accuracy = (TP + TN) /  TOTAL
* Recall = TP / (TP + FN)
* Precision = TP / (TP + FP)

---

### Regression Metrics

* Mean Absolute Error (MAE) -- **Not differentiable**
* Mean Squared Error (MSE)
* R2 Score = 1 - (MSE of the model / MSE of a horizontal/simple/benchmark model)
  * 0 -> Bad model
  * 1 -> Good model, due to MSE of the model being much smaller than that of benchmark model

---

### Types of Errors



* Under-fitting means the training set is not fitted well, and the **error due to bias**
* Over-fitting means the model is overcomplicated to memory the train set, and the **error due to variance**
* K-Fold Cross Validation

---

