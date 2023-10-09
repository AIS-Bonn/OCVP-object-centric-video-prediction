"""
Evaluating an object-centric predictor model checkpoint.
This module supports two different evaluations:
  - Visual quality of the predicted frames, pretty much video prediction.
  - Prediction quality of object dynamics. How well object segmentation masks are forecasted.
"""

import torch

from data import unwrap_batch_data, unwrap_batch_data_masks
from lib.arguments import get_predictor_evaluation_arguments
from lib.logger import Logger, print_
import lib.utils as utils

from base.basePredictorEvaluator import BasePredictorEvaluator


class Evaluator(BasePredictorEvaluator):
    """
    Evaluating an object-centric predictor model checkpoint.
    This module supports two different evaluations:
      - Visual quality of the predicted frames, pretty much video prediction.
      - Prediction quality of object dynamics. How well object segmentation masks are forecasted.
    """

    MODES = ["VideoPred", "Masks"]

    @torch.no_grad()
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
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        video_length = self.exp_params["training_prediction"]["sample_length"]
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        # fetching and preparing data
        videos, targets, condition, initializer_data = self.unwrap_function(self.exp_params, batch_data)
        videos, targets = videos.to(self.device), targets.to(self.device)
        if condition is not None:
            print(condition.shape)
            condition = condition.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that #seed {num_context} + #preds {num_preds}")

        # encoding images into object-centric slots, and temporally aligning slots
        out_model = self.model(videos, num_imgs=video_length, **initializer_data)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model
        # predicting future slots
        pred_slots = self.predictor(slot_history) if condition is None else self.predictor(slot_history, condition)
        # decoding predicted slots into predicted frames
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)

        # selecting predictions and targets given evaluation mode, and computing evaluation metrics
        if self.evaluation_mode == "VideoPred":
            preds_eval = img_recons.view(B, num_preds, C, H, W).clamp(0, 1)
            targets_eval = targets[:, num_context:num_context+num_preds, :, :].clamp(0, 1)
        elif self.evaluation_mode == "Masks":
            pred_masks = pred_masks.reshape(B, num_preds, -1, H, W)
            preds_eval = torch.argmax(pred_masks, dim=2).squeeze(2)
            targets_eval = targets[:, num_context:num_context+num_preds, :, :]
        else:
            raise ValueError(f"{self.evaluation_mode = } not recognized in {Evaluator.MODES = }...")
        self.metric_tracker.accumulate(
                preds=preds_eval,
                targets=targets_eval
            )
        return

    def set_evaluation_mode(self, evaluation_mode):
        """
        Toggling functions depending on the current evaluation model
        """
        if evaluation_mode not in Evaluator.MODES:
            raise ValueError(f"{evaluation_mode = } not recognized in {Evaluator.MODES = }...")
        print_(f"Setting evaluation to mode: {evaluation_mode}")

        if evaluation_mode == "VideoPred":
            self.evaluation_mode = "VideoPred"
            self.set_metric_tracker_video_pred()
            self.unwrap_function = unwrap_batch_data
        elif evaluation_mode == "Masks":
            self.evaluation_mode = "Masks"
            self.set_metric_tracker_object_pred()
            self.unwrap_function = unwrap_batch_data_masks
        return


if __name__ == "__main__":
    utils.clear_cmd()
    all_args = get_predictor_evaluation_arguments()
    exp_path, savi_model, checkpoint, name_predictor_experiment, args = all_args

    logger = Logger(exp_path=f"{exp_path}/{name_predictor_experiment}")
    logger.log_info("Starting object-centric predictor evaluation procedure", message_type="new_exp")
    print_("Initializing Evaluator...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")

    evaluator = Evaluator(
            name_predictor_experiment=name_predictor_experiment,
            exp_path=exp_path,
            savi_model=savi_model,
            checkpoint=args.checkpoint,
            num_preds=args.num_preds
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and predictor and loading pretrained parameters")
    evaluator.load_model()
    evaluator.setup_predictor()

    # VIDEO PREDICTION EVALUATION
    print_("Starting video predictor evaluation")
    evaluator.set_evaluation_mode(evaluation_mode="VideoPred")
    evaluator.evaluate()

    # OBJECT DYNAMICS EVALUATION (only on datasets with segmentation labels)
    db_name = evaluator.exp_params["dataset"]["dataset_name"]
    if db_name not in ["MoviA", "MoviC"]:
        print_(f"Dataset {db_name} does not support 'Masks' evaluation...\n Finishing execution")
        exit()
    print_("Starting object-centric evaluation")
    evaluator.set_evaluation_mode(evaluation_mode="Masks")
    evaluator.evaluate()


#
