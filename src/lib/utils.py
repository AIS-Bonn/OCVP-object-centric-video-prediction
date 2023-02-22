"""
Utils methods for bunch of purposes, including
    - Reading/writing files
    - Creating directories
    - Timestamp
    - Handling tensorboard
"""

import os
import pickle
import shutil
import random
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib.logger import log_function
from CONFIG import CONFIG


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def load_pickle_file(path):
    """ Loading pickle file """
    with open(path, "rb") as a_file:
        data = pickle.load(a_file)
    return data


def save_pickle_file(path, data):
    """ Saving pickle file """
    with open(path, "wb") as file:
        pickle.dump(data, file)
    return


def clear_cmd():
    """Clearning command line window"""
    os.system('cls' if os.name == 'nt' else 'clear')
    return


@log_function
def create_directory(dir_path, dir_name=None):
    """
    Creating a folder in given path.
    """
    if(dir_name is not None):
        dir_path = os.path.join(dir_path, dir_name)
    if(not os.path.exists(dir_path)):
        os.makedirs(dir_path)
    return


def delete_directory(dir_path):
    """
    Deleting a directory and all its contents
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    return


def split_path(path):
    """ Splitting a path into a list containing the names of all directories to the path """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def timestamp():
    """
    Obtaining the current timestamp in an human-readable way
    """
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


@log_function
def log_architecture(model, exp_path, fname="model_architecture.txt"):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # getting all_params
    with open(savepath, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if(isinstance(layer, torch.nn.Module)):
            log_module(module=layer, exp_path=exp_path, fname=fname)
    return


def log_module(module, exp_path, fname="model_architecture.txt", append=True):
    """
    Printing architecture modules into a txt file
    """
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    # writing from scratch or appending to existing file
    if (append is False):
        with open(savepath, "w") as f:
            f.write("")
    else:
        with open(savepath, "a") as f:
            f.write("\n\n")

    # writing info
    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return


def press_yes_to_continue(message, key="y"):
    """ Asking the user for input to continut """
    if isinstance(message, (list, tuple)):
        for m in message:
            print(m)
    else:
        print(message)
    val = input(f"Press '{key}' to continue...")
    if(val != key):
        print("Exiting...")
        exit()
    return


def press_yes_or_no(message):
    """ Asking the user for input for yes or no """
    if isinstance(message, (list, tuple)):
        for m in message:
            print(m)
    else:
        print(message)
    val = ""
    while val not in ["y", "n"]:
        val = input("Press 'y' to for yes or 'n' for no...")
        if val not in ["y", "n"]:
            print(f"  Key {val} is not valid...")
    return val


def get_from_dict(params, key_list):
    """ Getting a value from a dictionary given a list with the keys to get there """
    for key in key_list:
        params = params[key]
    return params


def set_in_dict(params, key_list, value):
    """ Updating a dictionary value, indexed by a list of keys to get there """
    for key in key_list[:-1]:
        # params = params.setdefault(key, {})
        params = params[key]
    params[key_list[-1]] = value
    return


class TensorboardWriter:
    """
    Class for handling the tensorboard logger

    Args:
    -----
    logdir: string
        path where the tensorboard logs will be stored
    """

    def __init__(self, logdir):
        """ Initializing tensorboard writer """
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        """ Adding a scalar for plot """
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        """ Adding several values in one plot """
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_images(self, fig_name, img_grid, step):
        """ Adding a new step image to a figure """
        self.writer.add_images(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        """ Adding a whole new figure to the tensorboard """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        """ Logging model graph to tensorboard """
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        """
        Logging a bunch of losses into the Tensorboard. Logging each of them into
        its independent plot and into a joined plot
        """
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return

#
