# CustomerChurnPredictor

A production-ready MLOps pipeline for predicting customer churn in e-commerce. Features include advanced data ingestion with RFM feature engineering, churn prediction using XGBoost, and a Flask API for serving predictions.

## Installation
1. Clone the repository: `git clone <repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure data: Update `config/data_config.yaml` with source details
4. Run the API: `python src/api/app.py`
5. Train the model: `python src/pipeline/train.py`
