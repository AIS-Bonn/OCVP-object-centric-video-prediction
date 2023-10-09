"""
Training and Validating a SAVi video decomposition model
"""

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # for avoiding memory leak
import torch

from data.load_data import unwrap_batch_data
from lib.arguments import get_directory_argument
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_recons

from base.baseTrainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Class for training a SAVi model for object-centric video
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
        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)

        # forward pass
        videos, targets = videos.to(self.device), targets.to(self.device)
        out_model = self.model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        if inference_only:
            return out_model, None

        # if necessary, doing loss computation, backward pass, optimization, and computing metrics
        self.loss_tracker(
                pred_imgs=reconstruction_history.clamp(0, 1),
                target_imgs=targets.clamp(0, 1)
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_slots"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training_slots"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    @torch.no_grad()
    def visualizations(self, batch_data, epoch, iter_):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.
        """
        if(iter_ % self.exp_params["training_slots"]["image_log_frequency"] != 0):
            return

        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model
        N = min(10, videos.shape[1])  # max of 10 frames for sleeker figures

        # output reconstructions versus targets
        visualize_recons(
                imgs=targets[0][:N],
                recons=reconstruction_history[0][:N].clamp(0, 1),
                tag="target",
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

        # output reconstructions and input images
        visualize_recons(
                imgs=videos[0][:N],
                recons=reconstruction_history[0][:N].clamp(0, 1),
                savepath=None,
                tb_writer=self.writer,
                iter=iter_
            )

        # Rendered individual objects
        fig, _, _ = visualize_decomp(
                individual_recons_history[0][:N].clamp(0, 1),
                savepath=None,
                tag="objects_decomposed",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        plt.close(fig)

        # Rendered individual object masks
        fig, _, _ = visualize_decomp(
                masks_history[0][:N].clamp(0, 1),
                savepath=None,
                tag="masks",
                cmap="gray",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_,
            )
        plt.close(fig)

        # Rendered individual combination of an object with its masks
        recon_combined = masks_history[0][:N] * individual_recons_history[0][:N]
        fig, _, _ = visualize_decomp(
                recon_combined.clamp(0, 1),
                savepath=None,
                tag="reconstruction_combined",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        plt.close(fig)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting SAVi training procedure", message_type="new_exp")

    print_("Initializing SAVi Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            exp_path=exp_path,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Setting up model and optimizer")
    trainer.setup_model()
    print_("Loading dataset...")
    trainer.load_data()
    print_("Starting to train")
    trainer.training_loop()


#
