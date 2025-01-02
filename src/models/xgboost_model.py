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
