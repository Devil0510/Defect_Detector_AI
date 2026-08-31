import logging
import sys
from pathlib import Path

def get_logger(name: str = "steel_defect", log_file: Path | str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Creates and configures a standardized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
