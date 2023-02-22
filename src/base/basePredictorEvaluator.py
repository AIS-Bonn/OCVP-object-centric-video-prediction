"""
Base predictor evaluator from which all predictor evaluator classes inherit.
Basically it removes the scaffolding that is repeat across all predictor evaluator modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib
from models.model_utils import freeze_params


@for_all_methods(log_function)
class BasePredictorEvaluator:
    """
    Base Class for evaluating a slot predictor model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    name_predictor_experiment: string
        Name of the predictor experiment (subdirectory in parent directory) to train.
    savi_model: string
        Name of the pretrained SAVI model used to extract object representation from frames
        and to decode the predicted slots back to images
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the experiment directory.
        If given, the model is initialized with the parameters of such checkpoint.
        This can be used to continue training or for transfer learning.
    num_preds: int
        Number of predictions to make and evaluate
    """

    def __init__(self, name_predictor_experiment, exp_path, savi_model, checkpoint,
                 num_preds=None, **kwargs):
        """
        Initializing the predictor evaluator object
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_predictor_experiment)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_model = savi_model
        self.checkpoint = checkpoint
        self.num_preds_args = num_preds

        # overriding 'num_preds' if given as argument
        if num_preds is not None:
            num_seed = self.exp_params["training_prediction"]["num_context"]
            self.exp_params["training_prediction"]["num_preds"] = num_preds
            self.exp_params["training_prediction"]["sample_length"] = num_seed + num_preds
            print_(f"  --> Overriding 'num_preds' to {num_preds}")
            print_(f"  --> New 'sample_length' is {num_seed + num_preds}")

        self.plots_path = os.path.join(self.exp_path, "plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)

        return

    def set_metric_tracker(self):
        """
        Initializing the metric tracker with the Video Prediction tracker by default
        """
        self.set_metric_tracker_video_pred()

    def set_metric_tracker_video_pred(self):
        """
        Initializing the metric tracker with Video Prediction metrics
        """
        self.metric_tracker = MetricTracker(
                self.exp_path,
                metrics=["psnr", "ssim", "lpips"]
            )

    def set_metric_tracker_object_pred(self):
        """
        Initializing the metric tracker with Object-centric metrics
        """
        self.test_set.get_masks = True
        self.metric_tracker = MetricTracker(
                self.exp_path,
                metrics=["segmentation_ari", "IoU"]
            )

    def load_data(self):
        """
        Loading test dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training_prediction"]["batch_size"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        self.test_set = datalib.load_data(
                exp_params=self.exp_params,
                split="test"
            )
        self.test_loader = datalib.build_data_loader(
                dataset=self.test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def load_model(self):
        """
        Load pretrained SAVi model from checkpoint
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        self.model = self.model.eval().to(self.device)

        # loading pretrained parameters and freezing
        checkpoint_path = os.path.join(self.parent_exp_path, "models", self.savi_model)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        freeze_params(self.model)
        return

    def setup_predictor(self):
        """
        Load pretrained predictor model from the corresponding model checkpoint
        """
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        predictor = predictor.eval().to(self.device)

        print_(f"Loading pretrained parameters from checkpoint {self.checkpoint}...")
        predictor = setup_model.load_checkpoint(
                checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                model=predictor,
                only_model=True,
            )
        self.predictor = predictor
        self.set_metric_tracker()
        return

    @torch.no_grad()
    def evaluate(self, save_results=True):
        """
        Evaluating model epoch loop
        """
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        self.model = self.model.eval()
        progress_bar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))

        # iterating test set and accumulating the results
        for i, batch_data in progress_bar:
            self.forward_eval(batch_data=batch_data)
            progress_bar.set_description(f"Iter {i}/{len(self.test_loader)}")

        self.metric_tracker.aggregate()
        _ = self.metric_tracker.summary()
        fname = f"{self.checkpoint[:-4]}_NumPreds={num_preds}"
        self.metric_tracker.save_results(exp_path=self.exp_path, fname=fname)
        self.metric_tracker.make_plots(
                start_idx=num_context,
                savepath=os.path.join(self.exp_path, "results", fname)
            )
        return

    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        """
        raise NotImplementedError("Base Evaluator Module does not implement 'forward_eval'...")


#
