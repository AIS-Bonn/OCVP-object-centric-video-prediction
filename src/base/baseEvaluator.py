"""
Base evaluator from which all backbone evaluator modules inherit.

Basically it removes the scaffolding that is repeat across all evaluation modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data


@for_all_methods(log_function)
class BaseEvaluator:
    """
    Base Class for evaluating a model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint to evaluate.
        It must be stored in the models/ directory of the experiment directory.
    """

    def __init__(self, exp_path, checkpoint):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        model_name = checkpoint.split(".")[0]
        self.results_name = f"{model_name}"

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        return

    def set_metric_tracker(self):
        """
        Initializing the metric tracker with evaluation metrics to track
        """
        self.metric_tracker = MetricTracker(
                self.exp_path,
                metrics=["segmentation_ari"]
            )

    def load_data(self):
        """
        Loading test-set and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = 1  # self.exp_params["training"]["batch_size"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        self.test_set = data.load_data(
                exp_params=self.exp_params,
                split="test"
            )
        self.test_loader = data.build_data_loader(
                dataset=self.test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def setup_model(self):
        """
        Initializing model and loading pretrained parameters given checkpoint
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        self.model = self.model.eval().to(self.device)

        # loading pretrained paramters
        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        self.set_metric_tracker()
        return

    @torch.no_grad()
    def evaluate(self, save_results=True):
        """
        Evaluating model
        """
        self.model = self.model.eval()
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))

        # iterating test set and accumulating the results
        for i, batch_data in progress_bar:
            self.forward_eval(batch_data=batch_data)
            progress_bar.set_description(f"Iter {i}/{len(self.test_loader)}")

        # computing average results and saving to results file
        self.metric_tracker.aggregate()
        self.results = self.metric_tracker.summary()
        if save_results:
            self.metric_tracker.save_results(exp_path=self.exp_path, fname=self.results_name)
        return

    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        raise NotImplementedError("Base Evaluator Module does not implement 'forward_eval'...")

#
