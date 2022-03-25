from configparser import ConfigParser


class MyConf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)
        self.add_sec = "Additional"

    def optionxform(self, optionstr):
        return optionstr


class Configurable(MyConf):
    def __init__(self, config_file):
        super().__init__()

        config = MyConf()
        config.read(config_file)
        self._config = config
        self.config_file = config_file

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)

        config.write(open(config_file, 'w'))

    def add_args(self, key, value):
        self._config.set(self.add_sec, key, value)
        self._config.write(open(self.config_file, 'w'))

    
    # # main
    # @property
    # def batch_size(self):
    #     return self._config.getint("main", "batch_size")

    # @property
    # def train_iter(self):
    #     return self._config.getint("main", "train_iter")

    # @property
    # def test_iter(self):
    #     return self._config.getint("main", "test_iter")

    # @property
    # def max_length(self):
    #     return self._config.getint("main", "max_length")

    # @property
    # def learning_rate(self):
    #     return self._config.getfloat("main", "learning_rate")

    # @property
    # def grad_iter(self):
    #     return self._config.getint("main", "grad_iter")

    # @property
    # def random_seed(self):
    #     return self._config.getint("main", "random_seed")


    # # continuum
    # @property
    # def continuum_dir(self):
    #     return self._config.get("continuum", "continuum_dir")

    # @property
    # def num_episode(self):
    #     return self._config.getint("continuum", "num_episode")

    # # few-nerd
    # @property
    # def supervised_train_file(self):
    #     return self._config.get("few-nerd", "supervised_train_file")

    # @property
    # def supervised_dev_file(self):
    #     return self._config.get("few-nerd", "supervised_dev_file")
    
    # @property
    # def supervised_test_file(self):
    #     return self._config.get("few-nerd", "supervised_test_file")
    
    # @property
    # def inter_train_file(self):
    #     return self._config.get("few-nerd", "inter_train_file")
    
    # @property
    # def inter_dev_file(self):
    #     return self._config.get("few-nerd", "inter_dev_file")
    
    # @property
    # def inter_test_file(self):
    #     return self._config.get("few-nerd", "inter_test_file")
    
    # @property
    # def intra_train_file(self):
    #     return self._config.get("few-nerd", "inter_train_file")
    
    # @property
    # def inter_dev_file(self):
    #     return self._config.get("few-nerd", "inter_dev_file")
    
    # @property
    # def inter_test_file(self):
    #     return self._config.get("few-nerd", "inter_test_file")

    # @property
    # def episode_inter_dir(self):
    #     return self._config.get("few-nerd", "episode_inter_dir")
    
    # @property
    # def episode_intra_dir(self):
    #     return self._config.get("few-nerd", "episode_intra_dir")

    
    # # stackoverflow
    # @property
    # def label_file(self):
    #     return self._config.get("stackoverflow-nerd", "label_file")
    
    # @property
    # def temporal_dir(self):
    #     return self._config.get("stackoverflow-nerd", "temporal_dir")
    
    # @property
    # def skewed_dir(self):
    #     return self._config.get("stackoverflow-nerd", "skewed_dir")






