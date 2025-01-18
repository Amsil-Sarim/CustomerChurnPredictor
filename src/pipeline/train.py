from src.data.ingest import CustomerDataIngestion
from src.models.xgboost_model import XGBoostChurnModel
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_churn_model(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ingestor = CustomerDataIngestion(config['data_config'])
    data = ingestor.load_data(config['data_path'])
    
    from sklearn.model_selection import train_test_split
    features = ['recency', 'frequency', 'monetary']
    X = data[features]
    y = data[config['target_column']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBoostChurnModel(config['model_params'])
    model.train(X_train, y_train, X_val, y_val)
    
    model.save(config['model_save_path'])
    logger.info(f"Churn prediction model saved to {config['model_save_path']}")

if __name__ == "__main__":
    train_churn_model("config/train_config.yaml")

def balance_dataset(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Balance dataset by undersampling majority class."""
    from sklearn.utils import resample
    majority = data[data[target_col] == 0]
    minority = data[data[target_col] == 1]
    majority_downsampled = resample(majority, n_samples=len(minority), random_state=42)
    balanced = pd.concat([majority_downsampled, minority])
    logger.info(f"Balanced dataset: {len(balanced)} samples")
    return balanced

def cross_validate_model(model, X, y, folds: int = 5):
    """Perform k-fold cross-validation for churn model."""
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model.model, X, y, cv=folds, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    logger.info(f"Cross-validation: Mean accuracy = {mean_score:.4f}, Std = {std_score:.4f}")
    return mean_score, std_score

def feature_selection(X, y, k: int = 3):
    """Select top-k features for churn prediction."""
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    logger.info(f"Selected features: {selected_features}")
    return selected_features

def save_model_checkpoint(model, epoch: int, path: str):
    """Save model checkpoint with epoch metadata."""
    import os
    checkpoint_path = f"{path}_epoch_{epoch}"
    model.save(checkpoint_path)
    with open(f"{checkpoint_path}.meta", 'w') as f:
        f.write(f"Epoch: {epoch}\nTimestamp: {datetime.now()}")
    logger.info(f"Saved checkpoint at {checkpoint_path}")

def log_training_metrics(model, X_val, y_val):
    """Log validation metrics during training."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    }
    logger.info(f"Training metrics: {metrics}")
    return metrics
