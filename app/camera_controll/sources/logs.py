from pathlib import Path
import logging

LOGS_DIRECTORY = Path(__file__).parent.parent.joinpath("results")
LOGS_DIRECTORY.mkdir(exist_ok=True)

def get_logger(name: str, encoding:str="utf-8"):
    """Logger for module fast and simple.

    Args:
        name (str): Module name for logging file
        encoding (str): encoding message in file

    Examples:
        ```python
        from sources.logs import get_logger

        logger = get_logger("Hallower_serv")

        logger.info("hallo, system!!!")
        ```

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d %(levelname)-4s %(message)s')
    handler = logging.FileHandler((LOGS_DIRECTORY/name).with_suffix(".log"), encoding=encoding)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger