from flask import Flask, request, jsonify
import joblib
import logging
from src.utils.logging import setup_logging

app = Flask(__name__)
setup_logging()
logger = logging.getLogger(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']  # Expected: {'recency': int, 'frequency': int, 'monetary': float}
        features = [[data['recency'], data['frequency'], data['monetary']]]
        prediction = model.predict(features)
        return jsonify({'churn_prediction': int(prediction[0])})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
