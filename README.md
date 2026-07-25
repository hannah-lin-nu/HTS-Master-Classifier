# HTS Master Classifier

**NLP-Based Product Classification for International Trade**  
IE7500 Data Analytics Engineering | 2026  
Group 4: Hannah Lin, Dario Garza, Dwayne Ford, Clemence Umutoni 

## Overview

The HTS Master Classifier is an NLP-based multiclass classification project that predicts 4-digit Harmonized Tariff Schedule (HTS) headings from unstructured product descriptions found in trade documents. The project is designed as a trade compliance decision-support tool that can help reduce manual classification effort and support more consistent review of product descriptions.

The core NLP task is supervised multiclass text classification. The input is a natural-language product description, and the output is a predicted 4-digit HTS heading. Supporting tasks include text preprocessing, HTS code extraction, label encoding, TF-IDF feature extraction, model training, model evaluation, and inference.

## Project Objectives

The main objective is to build and evaluate a reusable machine learning workflow for HTS heading prediction.

Specific objectives include:

- Predict 4-digit HTS headings from product descriptions.
- Compare multiple NLP and machine learning models.
- Evaluate model performance using classification metrics.
- Identify the strongest model for the current milestone.
- Analyze model strengths, limitations, and common failure cases.
- Support future validation against the official HTS reference taxonomy.

## Dataset

The project uses a locally cached copy of the CROSS rulings HTS dataset. The original Hugging Face dataset is no longer publicly available, so the notebook relies on a local copy downloaded on June 2, 2026.

The notebook also uses `hts_2026_revision_10_csv.csv` as the HTS reference taxonomy.

Dataset summary:

- 18,254 original records
- 549 unique 4-digit HTS classes before filtering
- Product descriptions and HTS labels are parsed from the messages field
- Rare classes with fewer than 10 samples are removed
- Final modeling dataset: 17,378 records across 258 HTS classes
- Train/test split: 80/20 stratified split
- Approximate split: 13,902 training records and 3,476 held-out test records

## Repository Structure

Suggested repository structure:

```text
HTS-Master-Classifier/
│
├── notebooks/
│   └── HTS_Classifier.ipynb
│
├── report/
│   └── Milestone_3_Model_Evaluation_Report.docx
│
├── data/
│   ├── README.md
│   └── hts_2026_revision_10_csv.csv
│
├── outputs/
│   ├── figures/
│   └── model_results/
│
├── README.md

```

## Methodology

The notebook implements the following workflow:

1. Load and parse the locally cached CROSS rulings dataset.
2. Extract product descriptions, HTS codes, and reasoning text from the `messages` field.
3. Clean product descriptions using `clean_text()`.
4. Extract the primary HTS code from the HTS label field.
5. Convert detailed HTS codes into 4-digit HTS headings.
6. Encode 4-digit HTS labels using `LabelEncoder`.
7. Remove rare classes with fewer than 10 records.
8. Split the filtered dataset into training and testing sets using an 80/20 stratified split.
9. Convert cleaned product descriptions into TF-IDF features.
10. Train and evaluate four classification models.
11. Compare model performance using classification metrics.
12. Analyze errors and model limitations.

## Feature Engineering

The project uses `TfidfVectorizer` to convert product descriptions into numerical features.

Current TF-IDF configuration:

- `max_features=50000`
- `ngram_range=(1, 2)`
- `min_df=2`
- Final TF-IDF feature space: 27,534 features

TF-IDF is appropriate for this task because HTS classification often depends on product-specific terms such as material, product type, technical use, chemical identifiers, and intended function.

## Models Evaluated

Four completed models are trained and evaluated using the same train-test split and TF-IDF feature representation:

1. **Logistic Regression + TF-IDF**
2. **Linear SVM + TF-IDF**
3. **Naive Bayes + TF-IDF**
4. **Feedforward Neural Network (FFNN) + TF-IDF**

## Evaluation Metrics

Because HTS heading prediction is a supervised multiclass classification problem, the notebook uses classification metrics rather than language-generation metrics.

The four tracked metrics are:

- **Accuracy:** Share of product descriptions assigned the correct 4-digit HTS heading.
- **Weighted Precision:** Precision averaged across classes and weighted by class support.
- **Weighted Recall:** Recall averaged across classes and weighted by class support.
- **Weighted F1-score:** Harmonic mean of weighted precision and weighted recall.

Weighted F1-score is the primary model-selection metric because it balances false positives and false negatives across an imbalanced 258-class label set.

## Model Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 54.1% | 56.4% | 54.1% | 50.1% |
| Linear SVM | 68.1% | 69.4% | 68.1% | 67.6% |
| Naive Bayes | 28.9% | 37.5% | 28.9% | 23.2% |
| FFNN | 63.8% | 66.9% | 63.8% | 63.3% |

## Model Comparison

Linear SVM is the best-performing model overall, achieving the highest accuracy, weighted precision, weighted recall, and weighted F1-score. It performs well because HTS product descriptions contain many keyword-based signals, and Linear SVM is effective with high-dimensional sparse TF-IDF features.

The FFNN is the second-best model, showing that neural network methods can perform well on TF-IDF features. However, it did not outperform the simpler and more efficient Linear SVM. Logistic Regression remains a useful fast and interpretable baseline. Naive Bayes performed weakest, likely due to class imbalance, the large number of HTS classes, and overlapping vocabulary across headings.

## Stability Check

The winning Linear SVM model was retrained and re-tested using five random seeds: `42`, `1`, `7`, `123`, and `2026`.

Observed results:

- Accuracy range: **67.4% to 68.5%**
- Weighted F1 range: **66.8% to 68.0%**
- Mean accuracy: **67.9%**
- Mean weighted F1: **67.5%**

This suggests that Linear SVM performance is stable across different train-test splits and is not dependent on one favorable split.

## Functional Evaluation

The notebook validates the internal end-to-end modeling workflow. It successfully loads and parses the data, cleans product descriptions, extracts HTS labels, converts HTS codes into 4-digit headings, filters rare classes, vectorizes text using TF-IDF, trains four models, generates predictions, and compares performance.

The notebook also includes:

- Model comparison table
- Bar chart comparing accuracy, precision, recall, and F1-score
- Confusion matrix for the best-performing model, Linear SVM
- Error analysis of misclassified test descriptions

A full user-facing prediction function has not yet been implemented. Future functional testing should include:

- Accepting a new product description as raw text
- Cleaning the input consistently with training data
- Applying the trained vectorizer
- Returning a predicted 4-digit HTS heading
- Returning confidence or score output where available
- Returning top-k candidate HTS headings for analyst review
- Validating predictions against the HTS reference file
- Handling blank, short, or unclear descriptions
- Measuring prediction latency

## Error Analysis

The Linear SVM model produced 1,108 misclassified test descriptions. These errors were grouped into two categories:

- **Near-miss errors:** 628 cases, or 57%, where the predicted heading was in the same general product category as the correct heading.
- **Unrelated-category errors:** 480 cases, or 43%, where the prediction was in a different product category.

Near-miss errors often occurred when products belonged to closely related headings that share similar vocabulary. Examples include apparel categories where the model identified the correct general area but confused similar headings.

The model also struggled with bundled or multi-part products. In these cases, the description may include several components, causing the model to focus on the most common or most visible term rather than the correct primary product classification.

## Key Insights

Linear SVM is the strongest current model because it combines strong predictive performance, stability, and relatively low computational cost. Its use of balanced class weighting helps address the uneven class distribution across the 258 HTS headings.

The FFNN showed strong performance but appeared more prone to overfitting. Its training loss dropped very low while test performance leveled off below SVM, suggesting that the model may have memorized training patterns more than it learned generalizable relationships.

TF-IDF performs well for this project, but it still depends on exact word matches. Product descriptions using synonyms, unusual phrasing, or unfamiliar terminology may still be misclassified.

## Current Status

Completed:

- Dataset loading and parsing
- Product description cleaning
- HTS code extraction
- Conversion to 4-digit HTS headings
- Rare-class filtering
- Train-test split
- TF-IDF vectorization
- Logistic Regression model
- Linear SVM model
- Naive Bayes model
- FFNN model
- Model evaluation and comparison
- Stability testing for Linear SVM
- Confusion matrix analysis
- Error analysis

## Limitations

This project should be treated as a decision-support tool, not an official customs ruling system. HTS classification can require legal interpretation, product-specific context, and regulatory review. The model predicts likely 4-digit headings based on historical product descriptions, but final classification decisions should be reviewed by a qualified trade compliance analyst.

Key limitations include:

- Class imbalance across HTS headings
- Reduced scope at the 4-digit heading level
- Dependence on product description quality
- Limited handling of rare HTS classes
- No current user-facing prediction interface
- TF-IDF dependence on exact wording
- Potential overfitting in the FFNN model
- Possible dataset bias from customs ruling examples

