"""
Creating a predictor experiment inside an existing experiment directory and initializing
its experiment parameters.
"""

import os
from lib.arguments import create_predictor_experiment_arguments
from lib.config import Config
from lib.logger import Logger, print_
from lib.utils import create_directory, delete_directory, clear_cmd
from CONFIG import CONFIG


def initialize_predictor_experiment():
    """
    Creating predictor experiment directory and initializing it with defauls
    """
    # reading command line args
    args = create_predictor_experiment_arguments()
    exp_dir, exp_name, predictor_name = args.exp_directory, args.name, args.predictor_name

    # making sure everything adds up
    parent_path = os.path.join(CONFIG["paths"]["experiments_path"], exp_dir)
    exp_path = os.path.join(CONFIG["paths"]["experiments_path"], exp_dir, exp_name)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"{parent_path = } does not exist")
    if not os.path.exists(os.path.join(parent_path, "experiment_params.json")):
        raise FileNotFoundError(f"{parent_path = } does not have experiment_params...")
    if len(os.listdir(os.path.join(parent_path, "models"))) <= 0:
        raise FileNotFoundError("Parent models-dir does not contain any models!...")
    if os.path.exists(exp_path):
        raise ValueError(f"{exp_path = } already exists. Choose a different name!")

    # creating directories
    create_directory(exp_path)
    _ = Logger(exp_path)  # initialize logger once exp_dir is created
    create_directory(dir_path=exp_path, dir_name="plots")
    create_directory(dir_path=exp_path, dir_name="tboard_logs")

    # adding experiment parameters from the parent directory, but only with specified predictor params
    try:
        cfg = Config(exp_path=parent_path)
        exp_params = cfg.load_exp_config_file()
        new_predictor_params = {}
        predictor_params = exp_params["model"]["predictor"]
        if predictor_name not in predictor_params.keys():
            raise ValueError(f"{predictor_name} not in keys {predictor_params.keys() = }")
        new_predictor_params["predictor_name"] = predictor_name
        new_predictor_params[predictor_name] = predictor_params[predictor_name]
        exp_params["model"]["predictor"] = new_predictor_params
        cfg.save_exp_config_file(exp_path=exp_path, exp_params=exp_params)
    except FileNotFoundError as e:
        print_("An error has occurred...\n Removing experiment directory")
        delete_directory(dir_path=exp_path)
        print(e)
    print(f"Predictor experiment {exp_name} created successfully! :)")
    return


if __name__ == "__main__":
    clear_cmd()
    initialize_predictor_experiment()

#
