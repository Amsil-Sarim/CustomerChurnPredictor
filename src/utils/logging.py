import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/app.log")
        ]
    )

def configure_advanced_logging(log_level: str = 'INFO', log_file: str = 'logs/app.log'):
    """Configure logging with rotation and custom format."""
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(getattr(logging, log_level))
    logger.info("Configured advanced logging with rotation")
