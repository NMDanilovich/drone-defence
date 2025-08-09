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
                config_dict[section][key] = float(value) if "." in value else int(value)
            except ValueError:
                config_dict[section][key] = value
                logging.warning(f"{key} is {type(value)}.")   
    
    
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

class BaseConfig:
    def __init__(self,  path: str = None, section_name: str = None):
        self.path = path
        self.data = read_config(path)
        
        if self.data is None:
            logging.warning("Config data is None!")
        elif len(self.data.items()) == 0:
            logging.warning("Lenght config data is 0!")

        if section_name is None:
            section = self.data
        else:
            self.section_name = section_name.upper()
            section = self.data[self.section_name]

        for key, value in section.items():
            setattr(self, key.upper(), value)

    def write(self):
        """Write the last config changes
        """

        for key, value in self.data[self.section_name].items():
            self.data[self.section_name][key] = self.__getattribute__(key.upper())
        
        write_config(self.path, self.data)


class CarriageConfig(BaseConfig):
    def __init__(self, path:str=None):
        super().__init__(
            path=Path(__file__).parent.joinpath("carriage.conf") if path is None else path, 
            section_name="carriage"
        )


class OverviewConfig(BaseConfig):
    def __init__(self, path:str=None):
        super().__init__(
            path=Path(__file__).parent.joinpath("overview.conf") if path is None else path, 
            section_name="overview"
        )

class TrackerConfig(BaseConfig):
    def __init__(self, path:str=None):
        super().__init__(
            path=Path(__file__).parent.joinpath("tracking.conf") if path is None else path, 
            section_name="tracking"
        )

class ConnactionsConfig(BaseConfig):
    def __init__(self, path:str=None):
        super().__init__(
            path=Path(__file__).parent.joinpath("connactions.conf") if path is None else path, 
        )

        self.NAMES = list(self.data.keys())

class CalibrationConfig(BaseConfig):
    def __init__(self, path:str=None):
        super().__init__(
            path=Path(__file__).parent.joinpath("calibration.conf") if path is None else path,
            section_name="calibration"
        )

        self.ALL = list(self.data[self.section_name].values())

__all__ = (CarriageConfig, OverviewConfig, TrackerConfig, ConnactionsConfig)
