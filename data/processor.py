import os
import data_config.config as data_config

CONFIG_FILE = "./data_config/config.cfg"


class Continuum():
    def __init__(self) -> None:
        pass


class DataProcessor():

    def __init__(self) -> None:
        self._config = data_config.Configurable(CONFIG_FILE)


    @property
    def fewnerd(self):
        if not os.path.exists('checkpoint'):
            os.mkdir('checkpoint')

    
    @property
    def stackoverflow(self):
        pass


    def _process_fewnerd(self):
        pass


    def _process_stackoverflow(self):
        pass


if __name__ == "__main__":
    pass