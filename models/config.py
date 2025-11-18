import yaml
import os

# read yaml file


def load_config():
    with open(os.path.join('models', 'config.yaml')) as file:
        config = yaml.safe_load(file)

    return config
