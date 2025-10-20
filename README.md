# Kaiburr Task 5: Consumer Complaint Classification

This repository contains the solution for Task 5 of the Kaiburr Full-Stack Assessment. The project involves building and evaluating several machine learning models to perform multi-class text classification on the Consumer Financial Protection Bureau (CFPB) dataset.

## Project Goal

The objective is to classify consumer complaint narratives into one of four product categories:
* Credit reporting, credit repair services, or other personal consumer reports
* Debt collection
* Consumer Loan
* Mortgage

## Methodology

The project follows a standard data science workflow:
1.  **Data Loading**: The dataset is loaded in chunks directly from the source URL to handle its large size and prevent memory errors.
2.  **Data Cleaning & Preprocessing**: The data is filtered to include only the four target categories. The complaint text is then cleaned by converting it to lowercase, removing punctuation and numbers, and filtering out common English stopwords.
3.  **Feature Engineering**: The cleaned text narratives are converted into numerical feature vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.
4.  **Model Training**: Three different classification models are trained on the preprocessed data:
    * Multinomial Naive Bayes
    * Logistic Regression
    * Linear Support Vector Machine (Linear SVM)
5.  **Model Evaluation**: Each model's performance is evaluated on a hold-out test set using a detailed classification report and a confusion matrix to identify its strengths and weaknesses.

---

## Model Performance Comparison

The final performance metrics for the three models are summarized below.

| Model                   | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| ----------------------- | -------- | ----------------- | -------------- | ---------------- |
| **Linear SVM** | **0.91** | **0.87** | **0.75** | **0.78** |
| **Logistic Regression** | **0.91** | 0.85              | 0.77           | 0.79             |
| Multinomial Naive Bayes | 0.87     | 0.75              | 0.72           | 0.73             |

### Conclusion

While both **Linear SVM** and **Logistic Regression** achieve the highest overall accuracy at **91%**, all models exhibit a significant challenge in correctly classifying the "Consumer Loan" category, likely due to overlapping language with other categories. Linear SVM shows the highest precision, making it slightly better at avoiding false positives.

---

## Detailed Model Results

### 1. Multinomial Naive Bayes

#### Classification Report
```text
                                                                        precision    recall  f1-score   support

                                                         Consumer Loan       0.43      0.26      0.33      1892
Credit reporting, credit repair services, or other personal consumer reports       0.90      0.92      0.91    161456
                                                       Debt collection       0.85      0.77      0.80     74326
                                                              Mortgage       0.80      0.95      0.87     26967

                                                              accuracy                           0.87    264641
                                                             macro avg       0.75      0.72      0.73    264641
                                                          weighted avg       0.87      0.87      0.87    264641


                                                                        precision    recall  f1-score   support

                                                         Consumer Loan       0.68      0.36      0.47      1892
Credit reporting, credit repair services, or other personal consumer reports       0.92      0.94      0.93    161456
                                                       Debt collection       0.87      0.83      0.85     74326
                                                              Mortgage       0.92      0.93      0.92     26967

                                                              accuracy                           0.91    264641
                                                             macro avg       0.85      0.77      0.79    264641
                                                          weighted avg       0.91      0.91      0.91    264641    


                                                                        precision    recall  f1-score   support

                                                         Consumer Loan       0.78      0.29      0.43      1892
Credit reporting, credit repair services, or other personal consumer reports       0.92      0.95      0.93    161456
                                                       Debt collection       0.88      0.83      0.85     74326
                                                              Mortgage       0.91      0.93      0.92     26967

                                                              accuracy                           0.91    264641
                                                             macro avg       0.87      0.75      0.78    264641
                                                          weighted avg       0.91      0.91      0.91    264641       

```
                                               