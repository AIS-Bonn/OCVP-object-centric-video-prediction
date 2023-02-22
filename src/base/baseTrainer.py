"""
Base trainer from which all backbone trainer classes inherit.
Basically it removes the scaffolding that is repeat across all training modules
"""

import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.loss import LossTracker
from lib.schedulers import WarmupVSScehdule
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib


@for_all_methods(log_function)
class BaseTrainer:
    """
    Base Class for training and validating a backbone model

    Args:
    -----
    exp_path: string
        Path to the experiment directory from which to read the experiment parameters,
        and where to store logs, plots and checkpoints
    checkpoint: string/None
        Name of a model checkpoint stored in the models/ directory of the experiment directory.
        If given, the model is initialized with the parameters of such checkpoint.
        This can be used to continue training or for transfer learning.
    resume_training: bool
        If True, saved checkpoint states from the optimizer, scheduler, ... are restored
        in order to continue training from the checkpoint
    """

    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = self.exp_params["training_slots"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        train_set = datalib.load_data(exp_params=self.exp_params, split="train")
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Examples in validation set: {len(valid_set)}")
        self.train_loader = datalib.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
        self.valid_loader = datalib.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        # loading optimizer, scheduler and loss
        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=model
            )
        loss_tracker = LossTracker(loss_params=self.exp_params["loss"])
        epoch = 0

        # loading pretrained model and other necessary objects for resuming training or fine-tuning
        if self.checkpoint is not None:
            print_(f"Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                model, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        self.model = model
        self.optimizer, self.scheduler, self.epoch = optimizer, scheduler, epoch
        self.loss_tracker = loss_tracker
        self.warmup_scheduler = WarmupVSScehdule(
                optimizer=self.optimizer,
                lr_warmup=lr_warmup,
                scheduler=scheduler
            )
        return

    @emergency_save
    def training_loop(self):
        """
        Repeating the process validation epoch - train epoch for the
        number of epoch specified in the exp_params file.
        """
        num_epochs = self.exp_params["training_slots"]["num_epochs"]
        save_frequency = self.exp_params["training_slots"]["save_frequency"]

        # iterating for the desired number of epochs
        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)

            # adding to tensorboard plot containing both losses
            self.writer.add_scalars(
                    plot_name='Total Loss',
                    val_names=["train_loss", "eval_loss"],
                    vals=[self.training_losses[-1], self.validation_losses[-1]],
                    step=epoch+1
                )

            # updating learning rate scheduler or lr-warmup
            self.warmup_scheduler(
                    iter=-1,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=True,
                    control_metric=self.validation_losses[-1]
                )

            # saving backup model checkpoint and, if reached saving frequency, epoch checkpoint
            self.wrapper_save_checkpoint(epoch=epoch, savename="checkpoint_last_saved.pth")
            if(epoch % save_frequency == 0 and epoch != 0):  # checkpoint_epoch_xx.pth
                print_("Saving model checkpoint")
                self.wrapper_save_checkpoint(epoch=epoch, savedir="models")

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        return

    def wrapper_save_checkpoint(self, epoch=None, savedir="models", savename=None, finished=False):
        """
        Wrapper for saving a models in a more convenient manner
        """
        setup_model.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.warmup_scheduler.scheduler,
                lr_warmup=self.warmup_scheduler.lr_warmup,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished
            )
        return

    def train_epoch(self, epoch):
        """
        Training epoch loop
        """
        self.loss_tracker.reset()
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, data in progress_bar:
            iter_ = len(self.train_loader) * epoch + i
            self.warmup_scheduler(iter=iter_, epoch=epoch, exp_params=self.exp_params, end_epoch=False)

            # forward pass, computing loss, backward pass, and update step
            out_model, loss = self.forward_loss_metric(
                    batch_data=data,
                    training=True
                )

            # logging and visualizations and other values
            self.visualizations(batch_data=data, epoch=epoch, iter_=iter_)
            if(iter_ % self.exp_params["training_slots"]["log_frequency"] == 0):
                self.writer.log_full_dictionary(
                        dict=self.loss_tracker.get_last_losses(),
                        step=iter_,
                        plot_name="Train Loss",
                        dir="Train Loss Iter",
                    )
                self.writer.add_scalar(
                        name="Learning Rate",
                        val=self.optimizer.param_groups[0]['lr'],
                        step=iter_
                    )

            # update progress bar
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Train Loss",
                dir="Train Loss",
            )
        self.training_losses.append(average_loss_vals["_total"].item())
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        """
        Validation epoch
        """
        self.loss_tracker.reset()
        progress_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        for i, data in progress_bar:
            _ = self.forward_loss_metric(
                    batch_data=data,
                    training=False
                )
            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        self.writer.log_full_dictionary(
                dict=average_loss_vals,
                step=epoch + 1,
                plot_name="Valid Loss",
                dir="Valid Loss",
            )
        self.validation_losses.append(average_loss_vals["_total"].item())
        return

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
            If True, only forward pass through the model is performed. Useful for generating images.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")

    def visualizations(self, batch_data, epoch, iter_):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, targets,
            or metadata, among others.
        epoch: int
            Number of the current training epoch.
        iter_: int
            Number of the current training iteration.
        """
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")


#
