# HTS Master Classifier

NLP-Based Product Classification for International Trade  
IE7500 Data Analytics Engineering | Group 4

## Overview

The HTS Master Classifier is an NLP-based multiclass classification project designed to predict 4-digit Harmonized Tariff Schedule (HTS) headings from unstructured product descriptions. The project supports trade compliance workflows by reducing the manual effort required to review product descriptions and identify likely HTS classifications.

HTS classification is a complex customs and trade compliance task. Incorrect classifications can result in shipment delays, tariff miscalculations, compliance risk, and possible penalties. This project explores whether supervised NLP models can assist analysts by recommending likely HTS headings from historical product descriptions and classification examples.

## Project Objective

The primary objective is to build and evaluate a reusable machine learning pipeline that can classify product descriptions into 4-digit HTS headings.

Specific objectives include:

- Predict the correct 4-digit HTS heading from product descriptions.
- Compare multiple NLP and machine learning models.
- Evaluate model performance using accuracy, weighted precision, weighted recall, and weighted F1-score.
- Build a reusable workflow for future prediction on new product descriptions.
- Support future integration with OCR-extracted text from invoices and trade documents.
- Support future validation against the official HTS reference taxonomy.

## NLP Task

This project is framed as a supervised multiclass text classification problem.

- Input: Natural-language product description
- Output: Predicted 4-digit HTS heading
- Supporting tasks: Text preprocessing, HTS extraction, label encoding, TF-IDF vectorization, model training, model evaluation, and model inference

## Dataset

The project uses a locally cached copy of the CROSS rulings HTS dataset. 

The notebook also uses `hts_2026_revision_10_csv.csv` as the HTS reference taxonomy.

Initial dataset summary:

- 18,254 records
- 549 unique 4-digit HTS classes before filtering
- Product descriptions and HTS labels are parsed from the `messages` field
- Rare classes with fewer than 10 samples are removed
- Final modeling dataset contains 17,378 records across 258 HTS classes

## Repository Structure

```text
HTS-Master-Classifier/
│
├── notebooks/
│   └── HTS_Classifier.ipynb
│
├── report/
│   └── Milestone_2_Model_Development_Report.docx
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
9. Convert product descriptions into TF-IDF features.
10. Train and evaluate four classification models.

## Feature Engineering

The project uses `TfidfVectorizer` to convert product descriptions into numerical features.

Current TF-IDF configuration:

- `max_features=50000`
- `ngram_range=(1, 2)`
- `min_df=2`
- Final TF-IDF feature space: 27,534 features

TF-IDF is appropriate for this task because HTS classification often depends on important product-specific terms such as material, product type, technical use, chemical identifiers, and intended function.

## Models Evaluated

Four models are trained and compared using the same train-test split and TF-IDF feature representation:

1. Logistic Regression
2. Linear Support Vector Machine
3. Multinomial Naive Bayes
4. Feedforward Neural Network

## Model Results

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 54.1% | 56.4% | 54.1% | 50.1% |
| Linear SVM | 68.1% | 69.4% | 68.1% | 67.6% |
| Naive Bayes | 28.9% | 37.5% | 28.9% | 23.2% |
| FFNN | 63.8% | 66.9% | 63.8% | 63.3% |

## Model Comparison

Linear SVM is the best-performing model overall, achieving the highest accuracy, weighted precision, weighted recall, and weighted F1-score. It is well suited for this task because HTS product descriptions contain many keyword-based signals, and Linear SVM performs effectively with high-dimensional sparse TF-IDF features.

The FFNN is the second-best model, showing that neural network methods can perform well on TF-IDF features. However, it did not outperform the simpler and more efficient Linear SVM. Logistic Regression remains a useful baseline because it is fast and interpretable. Naive Bayes performed the weakest, likely due to the large number of classes, class imbalance, and overlapping product vocabulary across HTS headings.

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
- Internal model evaluation and comparison

Pending:

- User-facing prediction function
- HTS reference validation
- OCR pipeline integration

## Limitations

This project should be treated as a decision-support tool, not an official customs ruling system. HTS classification can require legal interpretation, product-specific context, and regulatory review. The model predicts likely 4-digit headings based on historical product descriptions, but final classification decisions should be reviewed by a qualified trade compliance analyst.

Key limitations include:

- Class imbalance across HTS headings
- Reduced scope at the 4-digit heading level
- Dependence on product description quality
- Limited handling of rare HTS classes
- No current user-facing prediction interface
- No completed OCR integration yet
