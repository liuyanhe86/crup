from configparser import ConfigParser


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)
        self.add_sec = "Additional"

    def optionxform(self, optionstr):
        return optionstr


class Configurable(myconf):
    def __init__(self, config_file):
        super().__init__()

        config = myconf()
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

    # few-nerd
    @property
    def supervised_train_file(self):
        return self._config.get("few-nerd", "supervised_train_file")

    @property
    def supervised_dev_file(self):
        return self._config.get("few-nerd", "supervised_dev_file")
    
    @property
    def supervised_test_file(self):
        return self._config.get("few-nerd", "supervised_test_file")
    
    @property
    def inter_train_file(self):
        return self._config.get("few-nerd", "inter_train_file")
    
    @property
    def inter_dev_file(self):
        return self._config.get("few-nerd", "inter_dev_file")
    
    @property
    def inter_test_file(self):
        return self._config.get("few-nerd", "inter_test_file")
    
    @property
    def intra_train_file(self):
        return self._config.get("few-nerd", "inter_train_file")
    
    @property
    def inter_dev_file(self):
        return self._config.get("few-nerd", "inter_dev_file")
    
    @property
    def inter_test_file(self):
        return self._config.get("few-nerd", "inter_test_file")

    @property
    def episode_inter_dir(self):
        return self._config.get("few-nerd", "episode_inter_dir")
    
    @property
    def episode_intra_dir(self):
        return self._config.get("few-nerd", "episode_intra_dir")

    
    # stackoverflow
    @property
    def label_file(self):
        return self._config.get("stackoverflow-nerd", "label_file")
    
    @property
    def temporal_dir(self):
        return self._config.get("stackoverflow-nerd", "temporal_dir")
    
    @property
    def skewed_dir(self):
        return self._config.get("stackoverflow-nerd", "skewed_dir")






