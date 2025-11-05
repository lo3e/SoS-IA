import logging
import os

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)

        # StreamHandler per output su console
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # disattiva la propagazione ai logger root multipli
        logger.propagate = False

    return logger


# Se eseguito direttamente, test rapido
if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Test logger completato correttamente.")
