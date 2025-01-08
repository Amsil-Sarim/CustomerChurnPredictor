import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report

class XGBoostChurnModel:
    def __init__(self, params: dict):
        self.model = xgb.XGBClassifier(**params)
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)
        return accuracy, report
    
    def save(self, path):
        joblib.dump(self.model, path)

    def compute_feature_importance(self, X):
        """Calculate and log feature importance for churn prediction."""
        importance = self.model.feature_importances_
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_dict = dict(zip(feature_names, importance))
        logger.info(f"Feature importance: {importance_dict}")
        return importance_dict
