# backend/utils/logger.py

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with a consistent format.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Global logger for general usage
app_logger = get_logger("finance-advisor")
