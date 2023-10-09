"""
Evaluating a SAVI model checkpoint using object-centric metrics
This evalaution can only be performed on datasets with annotated segmentation masks
"""

import torch

from data import unwrap_batch_data_masks
from lib.arguments import get_sa_eval_arguments
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.utils as utils

from base.baseEvaluator import BaseEvaluator


@for_all_methods(log_function)
class Evaluator(BaseEvaluator):
    """
    Class for evaluating a SAVI model using object-centric metrics.
    This evalaution can only be performed on datasets with annotated segmentation masks
    """

    def set_metric_tracker(self):
        """
        Initializing the metric tracker
        """
        self.metric_tracker = MetricTracker(
                exp_path,
                metrics=["segmentation_ari", "IoU"]
            )
        return

    def load_data(self):
        """
        Loading data
        """
        super().load_data()
        self.test_set.get_masks = True
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
        videos, masks, _, initializer_kwargs = unwrap_batch_data_masks(self.exp_params, batch_data)
        videos, masks = videos.to(self.device), masks.to(self.device)
        out_model = self.model(
                videos,
                num_imgs=videos.shape[1],
                **initializer_kwargs
            )
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        # evaluation
        predicted_combined_masks = torch.argmax(masks_history, dim=2).squeeze(2)
        self.metric_tracker.accumulate(
                preds=predicted_combined_masks,
                targets=masks
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting SAVi object-cetric evaluation procedure", message_type="new_exp")

    print_("Initializing Evaluator...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    evaluator = Evaluator(
            exp_path=exp_path,
            checkpoint=args.checkpoint
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and loading pretrained parameters")
    evaluator.setup_model()
    print_("Starting evaluation")
    evaluator.evaluate()


#
