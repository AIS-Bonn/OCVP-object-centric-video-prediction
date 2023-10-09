"""
Evaluating a SAVI model checkpoint on a dataset without ground truth masks.
Since we do not have masks to compare with, we simply evaluate the visual qualitiy
of the reconstructed images using video-prediction metrics: PSNR, SSIM and LPIPS.
"""

from data.load_data import unwrap_batch_data
from lib.arguments import get_sa_eval_arguments
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.utils as utils

from base.baseEvaluator import BaseEvaluator


@for_all_methods(log_function)
class Evaluator(BaseEvaluator):
    """
    Evaluating a SAVI model checkpoint on a dataset without ground truth masks.
    Since we do not have masks to compare with, we simply evaluate the visual qualitiy
    of the reconstructed images using video-prediction metrics: PSNR, SSIM and LPIPS.
    """

    def set_metric_tracker(self):
        """
        Initializing the metric tracker
        """
        self.metric_tracker = MetricTracker(
                exp_path=exp_path,
                metrics=["psnr", "ssim", "lpips"]
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

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        videos, targets = videos.to(self.device), targets.to(self.device)
        out_model = self.model(
                videos,
                num_imgs=videos.shape[1],
                **initializer_kwargs
            )
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model
        self.metric_tracker.accumulate(
                preds=reconstruction_history.clamp(0, 1),
                targets=targets.clamp(0, 1)
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting SAVi visual quality evaluation procedure", message_type="new_exp")

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
