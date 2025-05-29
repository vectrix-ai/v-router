import logging
import os

import colorlog


def setup_logger(
    name: str,
    level: str = os.environ.get("LOG_LEVEL", "INFO"),
) -> logging.Logger:
    """Create a logger with colorized output."""
    logger = logging.getLogger(name)

    # Only set up handlers if none exist
    if not logger.hasHandlers():
        # Set logging level
        if level == "WARNING":
            logger.setLevel(logging.WARNING)
        elif level == "INFO":
            logger.setLevel(logging.INFO)
        elif level == "DEBUG":
            logger.setLevel(logging.DEBUG)
        elif level == "ERROR":
            logger.setLevel(logging.ERROR)
        else:
            raise ValueError(
                "Invalid logging level. Please choose from 'DEBUG', 'INFO', 'WARNING', or 'ERROR'."
            )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )
    logger.addHandler(console_handler)

    return logger
