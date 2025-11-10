"""
This module handles reading and writing of configuration files.

It provides functions to read and write .ini/.conf files and a base class
for managing different configuration types.
"""
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
                if "." in value:
                    config_dict[section][key] = float(value)
                elif "None" == value:
                    config_dict[section][key] = None
                elif "true" == value.lower():
                    config_dict[section][key] = True
                elif "false" == value.lower():
                    config_dict[section][key] = False
                elif value[0] in ("[(") and value[-1] in ("])"):
                    # delete brackets
                    if value[0] == "[":
                        value = value.strip("[]")
                    else:
                        value = value.strip("()")

                    value = tuple(int(num) for num in value.split(","))
                    config_dict[section][key] = value
                else:
                    config_dict[section][key] = int(value)

            except ValueError:
                config_dict[section][key] = value
    
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
    """
    A base class for handling configuration files.

    This class reads a configuration file, sets its sections as attributes,
    and provides a method to write changes back to the file.
    """
    def __init__(self,  path: str = None):
        """
        Initializes the BaseConfig.

        Args:
            path (str, optional): The path to the configuration file. Defaults to None.
        """
        self.path = path
        self.data = read_config(path)

        if str(self.path).find("example") != -1:
            logging.warning("Please use the copy of examples files!")
            write_config(self.path.replace(".example", ""), self.data)
        
        if self.data is None:
            logging.warning("Config data is None!")
        elif len(self.data.items()) == 0:
            logging.warning("Lenght config data is 0!")

        for key, value in self.data.items():
            setattr(self, key.upper(), value)

    def write(self):
        """
        Write the current configuration back to the file.
        """

        for key, value in self.data.items():
            self.data[key] = self.__getattribute__(key.upper())
        
        write_config(self.path, self.data)

class SystemConfig(BaseConfig):
    """A class for managing the system configuration."""
    def __init__(self, path:str=None):
        """
        Initializes the SystemConfig.

        Args:
            path (str, optional): The path to the system configuration file. 
                                  Defaults to 'system.conf' in the same directory.
        """
        super().__init__(
            path=Path(__file__).parent.joinpath("system.conf") if path is None else path, 
        )

class ConnactionsConfig(BaseConfig):
    """A class for managing the connections configuration."""
    def __init__(self, path:str=None):
        """
        Initializes the ConnactionsConfig.

        Args:
            path (str, optional): The path to the connections configuration file. 
                                  Defaults to 'connactions.conf' in the same directory.
        """
        super().__init__(
            path=Path(__file__).parent.joinpath("connactions.conf") if path is None else path, 
        )

        self.NAMES = list(self.data.keys())

__all__ = (SystemConfig, ConnactionsConfig)
