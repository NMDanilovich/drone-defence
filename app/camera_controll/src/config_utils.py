from dataclasses import dataclass
import logging
import configparser
from pathlib import Path

def read_config(path:str) -> dict:
    """Function for read ".ini" or ".conf" config file.

    Args:
        path (str): path to config

    Returns:
        dict: values of config file
                            e.g. {'SECTION': {'key': 'value'}}
    """

    config = configparser.ConfigParser()
    config.read(path)
    
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            try:
                # Try converting to integer first
                config_dict[section][key] = int(value)
            except ValueError:
                logging.error(f"Error of values in config. {key} is {type(value)}, but must be int.")   
    
    
    return config_dict

def write_config(path: str, config_data: dict):
    """Function for writing ".ini" or ".conf" config file.

    Args:
        path (str): path to config file
        config_data (dict): dictionary with config data,
                            e.g. {'SECTION': {'key': 'value'}}
    """
    config = configparser.ConfigParser()
    
    # configparser requires string values, so convert all values to strings
    stringified_config_data = {
        section: {key: str(value) for key, value in options.items()}
        for section, options in config_data.items()
    }
    
    config.read_dict(stringified_config_data)
            
    with open(path, 'w') as configfile:
        config.write(configfile)

@dataclass
class CarriageConfig:
    PATH = Path(__file__).parent.joinpath("carriage.conf")
    __DATA = read_config(PATH)

    START_X_POSITION = __DATA["START_POS"]["x_pos"]
    START_Y_POSITION = __DATA["START_POS"]["y_pos"]

    LAST_X_POSITION = __DATA["LAST_POS"]["x_pos"]
    LAST_Y_POSITION = __DATA["LAST_POS"]["y_pos"]

    MAX_Y_COORD = __DATA["LIMITS"]["max_y"]
    MIN_Y_COORD = __DATA["LIMITS"]["min_y"]
    MAX_X_COORD = __DATA["LIMITS"]["max_x"]
    MIN_X_COORD = __DATA["LIMITS"]["min_x"]

    del __DATA

    def write(self):
        """Write the last config changes
        """

        __DATA = {
            "START_POS": {
                "x_pos": self.START_X_POSITION,
                "y_pos": self.START_Y_POSITION
            },
            "LAST_POS": {
                "x_pos": self.LAST_X_POSITION,
                "y_pos": self.LAST_Y_POSITION
            },
            "LIMITS": {
                "max_x": self.MAX_X_COORD,
                "min_x": self.MIN_X_COORD,
                "max_y": self.MAX_Y_COORD,
                "min_y": self.MIN_Y_COORD,
            }
        }

        write_config(self.PATH, __DATA)