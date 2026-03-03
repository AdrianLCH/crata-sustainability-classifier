# Crata Sustainability Classifier

A text classification project that predicts whether a company is **sustainable (1)** or **not sustainable (0)** based on textual fields (e.g., `about`).

This repository includes:
- A training notebook with data cleaning and modeling experiments
- A saved scikit-learn pipeline (`joblib`)
- A minimal Flask API to serve predictions
- A Python client to query the API

## Approach (high level)

- Text preprocessing and cleaning
- TF-IDF vectorization
- Handling class imbalance using oversampling
- Model training and evaluation (Random Forest and Gradient Boosting)
- Persisting the best model pipeline as `Crata_model.pkl`

## Repository structure

```text
.
├── notebooks/
│   └── Data_Challenge.ipynb
├── models/
│   └── Crata_model.pkl
└── src/
    ├── api/
    │   └── app.py
    └── client/
        └── client.py
