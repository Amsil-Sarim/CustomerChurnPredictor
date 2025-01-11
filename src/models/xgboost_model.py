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

    def grid_search_params(self, X_train, y_train):
        """Perform grid search for hyperparameter tuning."""
        from sklearn.model_selection import GridSearchCV
        param_grid = {'max_depth': [3, 5, 7], 'learning_rate': [0.05, 0.1, 0.2]}
        grid = GridSearchCV(self.model, param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid.best_params_}")
        self.model.set_params(**grid.best_params_)

    def log_evaluation_metrics(self, X_test, y_test):
        """Log detailed evaluation metrics for the model."""
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_pred = self.predict(X_test)
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def apply_class_weights(self, y_train):
        """Apply class weights to handle imbalanced churn data."""
        from sklearn.utils.class_weight import compute_class_weight
        classes = [0, 1]
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))
        self.model.set_params(scale_pos_weight=weight_dict[1]/weight_dict[0])
        logger.info(f"Applied class weights: {weight_dict}")

    def save_with_version(self, path, version):
        """Save model with version metadata."""
        import os
        versioned_path = f"{path}_v{version}"
        self.save(versioned_path)
        with open(f"{versioned_path}.meta", 'w') as f:
            f.write(f"Version: {version}\nTimestamp: {datetime.now()}")
        logger.info(f"Saved model with version {version} at {versioned_path}")
