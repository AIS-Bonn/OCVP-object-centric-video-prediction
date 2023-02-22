""" Configs """

import os
from CONFIG import CONFIG


def get_available_configs():
    """ Getting a list with the name of the available config files """
    config_path = CONFIG["paths"]["configs_path"]
    files = sorted(os.listdir(config_path))
    available_configs = [f[:-5] for f in files if f[-5:] == ".json"]
    return available_configs
