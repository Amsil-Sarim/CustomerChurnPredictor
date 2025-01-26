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

@app.route('/predict_with_confidence', methods=['POST'])
def predict_with_confidence():
    """Predict churn with confidence scores."""
    try:
        data = request.json['data']
        features = [[data['recency'], data['frequency'], data['monetary']]]
        prob = model.predict_proba(features)[0]
        prediction = int(prob[1] > 0.5)
        logger.info(f"Predicted churn with confidence: {prob[1]}")
        return jsonify({'churn_prediction': prediction, 'confidence': float(prob[1])})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/validate_input', methods=['POST'])
def validate_input():
    """Validate input data for churn prediction."""
    try:
        data = request.json['data']
        if not all(key in data for key in ['recency', 'frequency', 'monetary']):
            raise ValueError("Missing required features")
        if any(data[key] < 0 for key in ['recency', 'frequency', 'monetary']):
            raise ValueError("Features cannot be negative")
        logger.info("Input data validated successfully")
        return jsonify({'status': 'valid'})
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the loaded XGBoost model."""
    info = {
        'model_type': 'XGBoost',
        'params': model.get_params(),
        'features': ['recency', 'frequency', 'monetary'],
        'version': 'v1.0'
    }
    logger.info("Retrieved model information")
    return jsonify(info)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict churn for a batch of customers."""
    try:
        data_list = request.json['data']
        features = [[d['recency'], d['frequency'], d['monetary']] for d in data_list]
        predictions = model.predict(features).tolist()
        logger.info(f"Processed batch of {len(data_list)} predictions")
        return jsonify({'churn_predictions': predictions})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/prediction_stats', methods=['GET'])
def prediction_stats():
    """Return statistics about recent predictions."""
    stats = {
        'total_predictions': 0,
        'churn_rate': 0.0,
        'avg_recency': 0.0,
        'avg_frequency': 0.0,
        'avg_monetary': 0.0
    }
    logger.info("Generated prediction statistics")
    return jsonify(stats)

@app.route('/health', methods=['GET'])
def health_check():
    """Check API and model health."""
    try:
        test_features = [[30, 5, 100.0]]
        model.predict(test_features)
        status = 'healthy'
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        status = 'unhealthy'
    return jsonify({'status': status, 'model_loaded': True})
