# Fake News Detection

-------------------------------------------------------------------
## To do
- Compute "linguistic cues" performances on FoR
- Cross performances "BERT" (GPU)
- Update the mini_paper

-------------------------------------------------------------------

This project replicates and discusses the findings from the paper "Exploring the Generalisability of Fake News Detection Models" by Nathaniel Hoy and Theodora Koulouri (2022). It evaluates the generalization of six traditional machine learning models across different preprocessing techniques and datasets.

## Rules
- You can work in small group (2 or 3 persons) but the report must be individual
- The deadline to submit your mini-project/project is May, 4th 2025
- Submit you report/code with this form : https://forms.gle/2YT2TcNtfaYZtogZ6

## Grading
- Adherence to guidelines and report structure, quality of writing: 5 points
- Relevance of data analysis: 3 points
- Relevance of state-of-the-art analysis: 3 points
- Relevance of the proposed model: 3 points
- Implementation of the model: 3 points
- Analysis of results: 3 points

## Objective

The goal is to evaluate how well six machine learning models (Logistic Regression, SVM, Random Forest, Gradient Boosting, AdaBoost, Neural Network) generalize to unseen data. We compare five preprocessing methods: Bag-of-Words (BoW), TF-IDF, Word2Vec, BERT and Liguistic Cues.

## Datasets

1. **ISOT Fake News Dataset**: A benchmark dataset with 44,898 articles, including 23,481 fake and 21,417 real news articles. It is used for training the models.
2. **Fake or Real News (FoR) Dataset**: An external dataset with 6,296 articles to test the generalization of the models.

## Models

- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- Gradient Boosting
- AdaBoost
- Neural Network (NN)

## Preprocessing Methods

- **BoW and TF-IDF**: Full text normalization.
- **Word2Vec, BERT & LC**: Lighter preprocessing for embedding-based models.

## Experiment

1. Train & test the models on the ISOT and FoR datasets.
2. Test their generalization by doing a cross-evaluation.
3. Compare model performance using accuracy, precision, recall, F1-score, and AUC.

## Results

Models trained on ISOT achieved near-perfect accuracy but showed performance drops on the external FoR dataset (and vice-verca), highlighting challenges in generalization.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/marcderoo/fake-news-detection.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Contributors
Chappuis Maxime & Deroo Marc

