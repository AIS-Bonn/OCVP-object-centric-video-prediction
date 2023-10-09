"""
Training and Validation of an object-centric predictor module using a frozen and pretrained
SAVI video decomposition model
"""
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # for avoiding memory leak
import torch

from data.load_data import unwrap_batch_data
from lib.arguments import get_predictor_training_arguments
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_qualitative_eval

from base.basePredictorTrainer import BasePredictorTrainer


class Trainer(BasePredictorTrainer):
    """
    Training and Validation of an object-centric predictor module using a frozen and pretrained
    SAVI video decomposition model
    """

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        """
        Computing a forwad pass through the model, and (if necessary) the loss values and metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        video_length = self.exp_params["training_prediction"]["sample_length"]
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        # fetching and checking data
        videos, targets, condition, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        videos, targets = videos.to(self.device), videos.to(self.device)
        if condition is not None:
            condition = condition.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that #seed {num_context} + #preds {num_preds}")

        # encoding frames into object slots usign pretrained SAVi
        with torch.no_grad():
            out_model = self.model(videos, num_imgs=video_length, **initializer_kwargs)
            slot_history, reconstruction_history, individual_recons_history, masks_history = out_model
        # predicting future slots
        pred_slots = self.predictor(slot_history) if condition is None else self.predictor(slot_history, condition)
        # rendering future objects and frames from predicted object slots
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)

        # Generating only model outputs
        out_model = (pred_imgs, pred_recons, pred_masks)
        if inference_only:
            return out_model, None

        # if necessary, doing loss computation, backward pass, optimization, and computing metrics
        target_slots = slot_history[:, num_context:num_context+num_preds, :, :]
        target_imgs = targets[:, num_context:num_context+num_preds, :, :]
        self.loss_tracker(
                preds=pred_slots,
                targets=target_slots,
                pred_imgs=pred_imgs,
                target_imgs=target_imgs
            )
        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_slots"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.predictor.parameters(),
                        self.exp_params["training_slots"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    @torch.no_grad()
    def visualizations(self, batch_data, epoch):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.
        """
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]

        # forward pass
        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        pred_imgs, pred_recons, pred_masks = out_model
        target_imgs = targets[:, num_context:num_context+num_preds, :, :]

        # visualitations
        ids = torch.linspace(0, videos.shape[0]-1, 3).round().int()  # equispaced videos in batch
        for idx in range(3):
            k = ids[idx]
            fig, ax = visualize_qualitative_eval(
                context=videos[k, :num_context],
                targets=target_imgs[k],
                preds=pred_imgs[k],
                savepath=None
            )
            self.writer.add_figure(tag=f"Qualitative Eval {k+1}", figure=fig, step=epoch + 1)
            plt.close(fig)

            objs = pred_masks[k*num_preds:(k+1)*num_preds] * pred_recons[k*num_preds:(k+1)*num_preds]
            fig, _, _ = visualize_decomp(
                    objs.clamp(0, 1),
                    savepath=None,
                    tag=f"Pred. Object Recons. {k+1}",
                    tb_writer=self.writer,
                    iter=epoch
                )
            plt.close(fig)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, savi_model, checkpoint, name_predictor_experiment, args = get_predictor_training_arguments()
    logger = Logger(exp_path=f"{exp_path}/{name_predictor_experiment}")
    logger.log_info("Starting object-centric predictor training procedure", message_type="new_exp")

    print_("Initializing Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            name_predictor_experiment=name_predictor_experiment,
            exp_path=exp_path,
            savi_model=savi_model,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model, predictor and optimizer")
    trainer.load_model()
    trainer.setup_predictor()
    print_("Starting to train")
    trainer.training_loop()


#
