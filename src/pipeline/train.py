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
