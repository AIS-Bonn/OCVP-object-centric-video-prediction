"""
Methods for processing command line arguments
"""

import os
import argparse

from configs import get_available_configs
from lib.utils import split_path

from CONFIG import CONFIG, PREDICTORS


def create_experiment_arguments():
    """
    Processing arguments for 01_create_experiment.py
    """
    configs = get_available_configs()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment"
                        "folder will be created", required=True, default="test_dir")
    parser.add_argument("--name", help="Name to give to the experiment")
    parser.add_argument("--config", help=f"Name of the predetermined 'config' to use: {configs}")
    args = parser.parse_args()

    args.config = check_config(args.config)
    return args


def create_predictor_experiment_arguments():
    """
    Processing arguments for 01_create_predictor_experiment.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Directory where the predictor experiment"
                        "will be created", required=True)
    parser.add_argument("--name", help="Name to give to the predictor experiment", required=True)
    parser.add_argument("--predictor_name", help=f"Name of the predictor module to use: {PREDICTORS}",
                        required=True)
    args = parser.parse_args()
    if args.predictor_name not in PREDICTORS:
        raise ValueError(f"Given predictor {args.predictor_name} not in allowed {PREDICTORS = }...")
    return args


def get_directory_argument():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint with pretrained parameters to load", default=None)
    parser.add_argument("--resume_training", help="For resuming training", default=False, action='store_true')
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint = process_checkpoint(exp_directory, args.checkpoint)
    return exp_directory, args


def get_sa_eval_arguments(get_num_preds=False):
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("-m", "--checkpoint", help="Checkpoint with pretrained parameters to load", default=None)
    if get_num_preds:
        parser.add_argument("--num_preds", help="""Number of rollout frames to predict for""",
                            type=int, default=None)
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint = process_checkpoint(exp_directory, args.checkpoint)
    return exp_directory, args


def get_predictor_training_arguments():
    """
    Processing arguments for predictor training script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the father exp. directory", required=True)
    parser.add_argument("--checkpoint", help="""Checkpoint with predictor pretrained parameters
                        to load""", default=None)
    parser.add_argument("--resume_training", help="Resuming training", default=False, action='store_true')
    parser.add_argument("-m", "--savi_model", help="""Path to SAVi checkpoint to be used during
                        in training or validation, from inside the experiments directory""",
                        required=True)
    parser.add_argument("--name_predictor_experiment", help="""Name to the directory inside the
                        exp_directory corresponding to a predictor experiment.""", required=True)
    args = parser.parse_args()
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.savi_model = process_checkpoint(exp_directory, args.savi_model)
    args.checkpoint = process_predictor_checkpoint(
            exp_path=exp_directory,
            name_predictor_experiment=args.name_predictor_experiment,
            checkpoint=args.checkpoint
        )

    return exp_directory, args.savi_model, args.checkpoint, args.name_predictor_experiment, args


def get_predictor_evaluation_arguments():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the father exp. directory", required=True)
    parser.add_argument("-m", "--savi_model", help="""Path to SAVi checkpoint to be used during
                        training or validation, from inside the experiments directory""",
                        required=True)
    parser.add_argument("--name_predictor_experiment", help="""Name to the directory inside the
                        exp_directory corresponding to a predictor experiment.""", required=True)
    parser.add_argument("--checkpoint", help="""Checkpoint with predictor pretrained parameters
                        to load for evalution""", required=True)
    parser.add_argument("--num_preds", help="""Number of rollout frames to predict for""",
                        type=int, default=None)

    args = parser.parse_args()
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.savi_model = process_checkpoint(exp_directory, args.savi_model)
    args.checkpoint = process_predictor_checkpoint(
            exp_path=exp_directory,
            name_predictor_experiment=args.name_predictor_experiment,
            checkpoint=args.checkpoint
        )

    return exp_directory, args.savi_model, args.checkpoint, args.name_predictor_experiment, args


def get_generate_figs_savi():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the father exp. directory", required=True)
    parser.add_argument("-m", "--savi_model", help="""Path to SAVi checkpoint to be used during
                        training or validation, from inside the experiments directory""",
                        required=True)
    parser.add_argument("--num_seqs", help="""Number of sequences to generate""",
                        type=int, default=30)

    args = parser.parse_args()
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.savi_model = process_checkpoint(exp_directory, args.savi_model)
    return exp_directory, args


def get_generate_figs_pred():
    """
    Processing arguments for predictor evaluation script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the father exp. directory", required=True)
    parser.add_argument("-m", "--savi_model", help="""Path to SAVi checkpoint to be used during
                        training or validation, from inside the experiments directory""",
                        required=True)
    parser.add_argument("--name_predictor_experiment", help="""Name to the directory inside the
                        exp_directory corresponding to a predictor experiment.""", required=True)
    parser.add_argument("--checkpoint", help="""Checkpoint with predictor pretrained parameters
                        to load for evalution""", required=True)
    parser.add_argument("--num_preds", help="""Number of rollout frames to predict for""",
                        type=int, default=25)
    parser.add_argument("--num_seqs", help="""Number of sequences to generate""",
                        type=int, default=30)

    args = parser.parse_args()
    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.savi_model = process_checkpoint(exp_directory, args.savi_model)
    args.checkpoint = process_predictor_checkpoint(
            exp_path=exp_directory,
            name_predictor_experiment=args.name_predictor_experiment,
            checkpoint=args.checkpoint
        )
    return exp_directory, args


def process_experiment_directory_argument(exp_directory, create=False):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """
    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    split_exp_dir = split_path(exp_directory)
    if os.path.basename(exp_path) == split_exp_dir[0]:
        exp_directory = "/".join(split_exp_dir[1:])

    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory) and create is False):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()
    elif(not os.path.exists(exp_directory) and create is True):
        os.makedirs(exp_directory)

    return exp_directory


def check_config(config):
    """
    Making sure that the predetermined configuration file, if given, exists
    """
    if config is None or len(config) < 1:
        return None
    else:
        config_path = CONFIG["paths"]["configs_path"]
        if config[-5:] != ".json":
            config = f"{config}.json"
        if config is not None and not os.path.join(config_path, config):
            raise FileNotFoundError(f"Given config. {config} does not exist in configs. path {config_path}")
        return config


def process_checkpoint(exp_path, checkpoint):
    """
    Making sure checkpoint exists
    """
    if checkpoint is not None:
        checkpoint_path = os.path.join(exp_path, "models", checkpoint)
        print(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist in exp {exp_path}")
    return checkpoint


def process_predictor_checkpoint(exp_path, name_predictor_experiment,  checkpoint):
    """
    Making sure checkpoint exists
    """
    if checkpoint is not None:
        checkpoint_path = os.path.join(exp_path, name_predictor_experiment, "models", checkpoint)
        print(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist in exp {checkpoint_path}")
    return checkpoint


#
