import logging
from logging import getLogger,FileHandler,StreamHandler

def set_logger(logger_name:str) -> logging.Logger:
    logger = getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    s_handler = StreamHandler()
    s_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    s_handler.setFormatter(formatter)

    # before adding handler, check if there already exists
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(s_handler)
    return logger
