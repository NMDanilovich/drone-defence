from pathlib import Path
import logging

LOGS_DIRECTORY = Path(__file__).parent.parent.joinpath("logs")
LOGS_DIRECTORY.mkdir(exist_ok=True)

def get_logger(name: str, encoding:str="utf-8", directory=LOGS_DIRECTORY, terminal=True):
    """Logger for module fast and simple.

    Args:
        name (str): Module name for logging file
        encoding (str): encoding message in file, by default "utf-8".
        directory (str): path to logs directory, by default "drone-defence/app/camera_control/logs".
        terminal (bool): flag for terminal output data logger, by default True.

    Examples:
        ```python
        from sources.logs import get_logger

        logger = get_logger("Hallower")

        logger.info("hallo, system!!!")
        ```

    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)-6d %(levelname)-4s %(message)s')
    
    file_handler = logging.FileHandler((directory/name).with_suffix(".log"), encoding=encoding)
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    if terminal:
        cli_handler = logging.StreamHandler()
        cli_handler.setFormatter(formatter)
        logger.addHandler(cli_handler)
    
    return logger