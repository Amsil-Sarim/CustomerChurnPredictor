import pandas as pd
import great_expectations as ge
import logging
import yaml
from dvc.api import DVCFileSystem
from typing import Optional, Dict
from pathlib import Path
import boto3
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomerDataIngestion:
    def __init__(self, config_path: str):
        """Initialize with a configuration file for customer data ingestion."""
        self.config = self._load_config(config_path)
        self.s3_client = boto3.client('s3') if self.config.get('source') == 's3' else None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    def load_data(self, file_path: str, version: Optional[str] = None) -> pd.DataFrame:
        """Load customer data from local filesystem or S3 with versioning support."""
        try:
            if self.config.get('source') == 's3':
                data = self._load_from_s3(file_path)
            else:
                data = self._load_from_local(file_path, version)
            
            data = self._validate_data(data)
            data = self._compute_rfm_features(data)
            
            logger.info(f"Successfully loaded and processed customer data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load customer data: {str(e)}")
            raise

    def _load_from_local(self, file_path: str, version: Optional[str]) -> pd.DataFrame:
        """Load data from local filesystem with DVC versioning."""
        if version:
            fs = DVCFileSystem()
            local_path = Path(f"data/{Path(file_path).name}")
            fs.get(file_path, str(local_path), version=version)
            file_path = local_path
        return pd.read_csv(file_path)

    def _load_from_s3(self, file_path: str) -> pd.DataFrame:
        """Load data from S3 bucket."""
        bucket = self.config['s3_bucket']
        obj = self.s3_client.get_object(Bucket=bucket, Key=file_path)
        return pd.read_csv(obj['Body'])

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate customer data using Great Expectations."""
        ge_df = ge.from_pandas(data)
        expectations = [
            ge_df.expect_column_values_to_not_be_null(self.config['target_column']),
            ge_df.expect_column_mean_to_be_between(self.config['numeric_column'], min_value=0, max_value=1000)
        ]
        for expectation in expectations:
            if not expectation['success']:
                raise ValueError(f"Data validation failed: {expectation['exception_info']}")
        logger.info("Customer data passed all validation checks")
        return data

    def _compute_rfm_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute RFM (Recency, Frequency, Monetary) features for churn prediction."""
        current_date = pd.to_datetime(self.config['current_date'])
        data['order_date'] = pd.to_datetime(data['order_date'])
        
        # Recency: Days since last purchase
        data['recency'] = (current_date - data.groupby('customer_id')['order_date'].transform('max')).dt.days
        
        # Frequency: Number of purchases
        data['frequency'] = data.groupby('customer_id')['order_id'].transform('count')
        
        # Monetary: Total spend
        data['monetary'] = data.groupby('customer_id')['amount'].transform('sum')
        
        # Drop duplicates after grouping
        data = data.drop_duplicates(subset=['customer_id'])
        
        return data

if __name__ == "__main__":
    ingestor = CustomerDataIngestion("config/data_config.yaml")
    data = ingestor.load_data("data/customer_data.csv", version="v1.0")
    print(data.head())

    def compute_tenure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate customer tenure based on first order date."""
        data['first_order_date'] = data.groupby('customer_id')['order_date'].transform('min')
        current_date = pd.to_datetime(self.config['current_date'])
        data['tenure_days'] = (current_date - data['first_order_date']).dt.days
        data['tenure_years'] = data['tenure_days'] / 365.0
        logger.info("Computed customer tenure features")
        return data.drop(columns=['first_order_date'])

    def normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize RFM features to 0-1 range."""
        for col in ['recency', 'frequency', 'monetary']:
            if col in data.columns:
                data[f'{col}_norm'] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                data[f'{col}_norm'] = data[f'{col}_norm'].fillna(0)
        logger.info("Normalized RFM features to 0-1 range")
        return data

    def compute_avg_order_value(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate average order value per customer."""
        data['total_orders'] = data.groupby('customer_id')['order_id'].transform('count')
        data['total_spend'] = data.groupby('customer_id')['amount'].transform('sum')
        data['avg_order_value'] = data['total_spend'] / data['total_orders']
        data['avg_order_value'] = data['avg_order_value'].fillna(0)
        logger.info("Computed average order value feature")
        return data.drop(columns=['total_orders', 'total_spend'])

    def validate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate required columns for churn prediction."""
        required = ['customer_id', 'order_date', 'order_id', 'amount', self.config['target_column']]
        missing = [col for col in required if col not in data.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            raise ValueError(f"Required columns missing: {missing}")
        logger.info("Validated all required columns")
        return data

    def log_data_stats(self, data: pd.DataFrame) -> None:
        """Log statistics about the customer dataset."""
        stats = {
            'num_customers': len(data),
            'avg_recency': data['recency'].mean(),
            'avg_frequency': data['frequency'].mean(),
            'avg_monetary': data['monetary'].mean(),
            'churn_rate': data[self.config['target_column']].mean()
        }
        logger.info(f"Dataset stats: {stats}")

    def filter_invalid_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid customer records."""
        original_len = len(data)
        data = data.dropna(subset=['customer_id', 'order_date'])
        data = data[data['recency'] >= 0]
        data = data[data['monetary'] >= 0]
        logger.info(f"Filtered {original_len - len(data)} invalid records")
        return data.reset_index(drop=True)

    def compute_customer_segments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Segment customers based on RFM scores."""
        data['recency_score'] = pd.qcut(data['recency'], q=4, labels=False, duplicates='drop')
        data['frequency_score'] = pd.qcut(data['frequency'], q=4, labels=False, duplicates='drop')
        data['monetary_score'] = pd.qcut(data['monetary'], q=4, labels=False, duplicates='drop')
        data['rfm_segment'] = data[['recency_score', 'frequency_score', 'monetary_score']].sum(axis=1)
        logger.info("Computed RFM-based customer segments")
        return data
