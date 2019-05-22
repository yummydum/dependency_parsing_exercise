import logging
from logging import getLogger,FileHandler,StreamHandler
from pathlib import Path

def set_logger(logger_name:str) -> logging.Logger:
    logger = getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    s_handler = StreamHandler()
    s_handler.setLevel(logging.DEBUG)
    f_handler = FileHandler(Path("log",f"{logger_name}.log"))
    f_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # before adding handler, check if there already exists
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
    return logger
