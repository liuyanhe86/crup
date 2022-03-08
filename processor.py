import os
import config.config as data_config

CONFIG_FILE = "config/config.cfg"


class Continuum:
    def __init__(self, dataset:str, config:data_config.Configurable) -> None:
        self.path = os.path.join(config.continuum_dir, dataset)
        self.episode_num = config.num_episode
    
    def train_episodes(self):
        episode_files = os.listdir(os.path.join(self.path, 'train'))
        for file in episode_files:
            pass



class DataProcessor:

    def __init__(self) -> None:
        self._config = data_config.Configurable(CONFIG_FILE)


    @property
    def fewnerd(self):
        if not os.path.exists(self._config.continuum_dir):
            self._process_fewnerd()
        return Continuum(dataset="few-nerd", config=self._config)

    
    @property
    def stackoverflow(self):
        pass


    def _process_fewnerd(self):
        pass


    def _process_stackoverflow(self):
        pass


if __name__ == "__main__":
    pass