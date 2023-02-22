"""
Methods to manage parameters and configurations
TODO:
 - Add support to change any values from the command line
"""

import os
import json

from lib.logger import print_
from lib.utils import timestamp
from CONFIG import DEFAULTS, CONFIG


class Config(dict):
    """
    """
    _default_values = DEFAULTS
    _help = "Potentially you can add here comments for what your configs are"
    _config_groups = ["dataset", "model", "training", "loss"]

    def __init__(self, exp_path):
        """
        Populating the dictionary with the default values
        """
        for key in self._default_values.keys():
            self[key] = self._default_values[key]
        self["_general"] = {}
        self["_general"]["exp_path"] = exp_path
        return

    def create_exp_config_file(self, exp_path=None, config=None):
        """
        Creating a JSON file with exp configs in the experiment path
        """
        exp_path = exp_path if exp_path is not None else self["_general"]["exp_path"]
        if not os.path.exists(exp_path):
            raise FileNotFoundError(f"ERROR!: exp_path {exp_path} does not exist...")

        if config is not None:
            config_file = os.path.join(CONFIG["paths"]["configs_path"], config)
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Given config file {config_file} does not exist...")

            with open(config_file) as file:
                self = json.load(file)
                self["_general"] = {}
                self["_general"]["exp_path"] = exp_path
            print_(f"Creating experiment parameters file from config {config}...")

        self["_general"]["created_time"] = timestamp()
        self["_general"]["last_loaded"] = timestamp()
        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(self, file)
        return

    def load_exp_config_file(self, exp_path=None):
        """
        Loading the JSON file with exp configs
        """
        if exp_path is not None:
            self["_general"]["exp_path"] = exp_path
        exp_config = os.path.join(self["_general"]["exp_path"], "experiment_params.json")
        if not os.path.exists(exp_config):
            raise FileNotFoundError(f"ERROR! exp. configs file {exp_config} does not exist...")

        with open(exp_config) as file:
            self = json.load(file)
        self["_general"]["last_loaded"] = timestamp()
        return self

    def update_config(self, exp_params):
        """
        Updating an experiments parameters file with newly added configurations from CONFIG.
        """
        # TODO: Add recursion to make it always work
        for group in Config._config_groups:
            if not isinstance(Config._default_values[group], dict):
                continue
            for k in Config._default_values[group].keys():
                if(k not in exp_params[group]):
                    if(isinstance(Config._default_values[group][k], (dict))):
                        exp_params[group][k] = {}
                    else:
                        exp_params[group][k] = Config._default_values[group][k]

                if(isinstance(Config._default_values[group][k], dict)):
                    for q in Config._default_values[group][k].keys():
                        if(q not in exp_params[group][k]):
                            exp_params[group][k][q] = Config._default_values[group][k][q]
        return exp_params

    def save_exp_config_file(self, exp_path=None, exp_params=None):
        """
        Dumping experiment parameters into path
        """
        exp_path = self["_general"]["exp_path"] if exp_path is None else exp_path
        exp_params = self if exp_params is None else exp_params

        exp_config = os.path.join(exp_path, "experiment_params.json")
        with open(exp_config, "w") as file:
            json.dump(exp_params, file)
        return

#
