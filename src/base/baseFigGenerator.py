"""
Base module for generating images from a pretrained model.
All other Figure Generator modules inherit from here.

Basically it removes the scaffolding that is repeat across all Figure Generation modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import log_function, for_all_methods
import lib.setup_model as setup_model
import lib.utils as utils
import data
from models.model_utils import freeze_params


@for_all_methods(log_function)
class BaseFigGenerator:
    """
    Base Class for figure generation

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    savi_model: string/None
        Name of SAVI model checkpoint to use when generating figures.
        It must be stored in the models/ directory of the experiment directory.
    num_seqs: int
        Number of sequences to process and save
    """

    def __init__(self, exp_path, savi_model, num_seqs=10):
        """
        Initializing the figure generator object
        """
        self.exp_path = os.path.join(exp_path)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_model = savi_model
        self.num_seqs = num_seqs

        model_name = savi_model.split('.')[0]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_SaVIModel_{model_name}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = 1
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        # test_set = data.load_data(exp_params=self.exp_params, split="test")
        test_set = data.load_data(exp_params=self.exp_params, split="valid")
        self.test_set = test_set
        self.test_loader = data.build_data_loader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def load_model(self, exp_path=None):
        """
        Load pretraiened SAVi model from checkpoint

        Args:
        -----
        exp_path: sting/None
            If None, 'self.exp_path' is used. Otherwise, the given path is used to load
            the pretrained model paramters
        """
        # to use the same function for SAVi and Predictor figure generation
        exp_path = exp_path if exp_path is not None else self.exp_path

        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(exp_path, "models", self.savi_model)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        freeze_params(self.model)
        return

    def load_predictor(self):
        """
        Load pretrained predictor model from the corresponding model checkpoint
        """
        # loading model
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        predictor = predictor.eval().to(self.device)

        # loading pretrained predictor
        predictor = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=predictor,
                    only_model=True,
                )
        self.predictor = predictor
        return

    @torch.no_grad()
    def generate_figs(self):
        """
        Computing and saving visualizations
        """
        progress_bar = tqdm(enumerate(self.test_loader), total=self.num_seqs)
        for i, batch_data in progress_bar:
            if i >= self.num_seqs:
                break
            self.compute_visualization(batch_data=batch_data, img_idx=i)
        return

    def compute_visualization(self, batch_data, img_idx, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        img_idx: int
            Index of the visualization to compute and save
        """
        raise NotImplementedError("Base FigGenerator does not implement 'compute_visualization'...")

#
