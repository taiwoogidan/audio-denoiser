import logging

def get_logger() -> logging.Logger:
    logger = logging.getLogger("denoiserLog")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger