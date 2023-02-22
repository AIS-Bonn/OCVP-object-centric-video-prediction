"""
Loss functions and loss-related utils
"""

import torch
import torch.nn as nn
from lib.logger import log_info
from CONFIG import LOSSES


class LossTracker:
    """
    Class for computing, weighting and tracking several loss functions

    Args:
    -----
    loss_params: dict
        Loss section of the experiment paramteres JSON file
    """

    def __init__(self, loss_params):
        """
        Loss tracker initializer
        """
        assert isinstance(loss_params, list), f"Loss_params must be a list, not {type(loss_params)}"
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(f"Loss {loss['type']} not implemented. Use one of {LOSSES}")

        self.loss_computers = {}
        for loss in loss_params:
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
        self.reset()
        return

    def reset(self):
        """
        Reseting loss tracker
        """
        self.loss_values = {loss: [] for loss in self.loss_computers.keys()}
        self.loss_values["_total"] = []
        return

    def __call__(self, **kwargs):
        """
        Wrapper for calling accumulate
        """
        self.accumulate(**kwargs)

    def accumulate(self, **kwargs):
        """
        Computing the different metrics, weigting them according to their multiplier,
        and adding them to the results list.
        """
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            self.loss_values[loss].append(loss_val)
            total_loss = total_loss + loss_val * self.loss_computers[loss]["weight"]
        self.loss_values["_total"].append(total_loss)
        return

    def aggregate(self):
        """
        Aggregating the results for each metric
        """
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_computers:
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        """
        Fetching the last computed loss value for each loss function
        """
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        """
        Printing and fetching the results
        """
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val


def get_loss(loss_type="mse", **kwargs):
    """
    Loading a function of object for computing a loss
    """
    if loss_type not in LOSSES:
        raise NotImplementedError(f"Loss {loss_type} not available. Use one of {LOSSES}")

    print(f"creating loss function of type: {loss_type}")
    if loss_type in ["mse", "l2"]:
        loss = MSELoss()
    elif loss_type in ["pred_img_mse"]:
        loss = PredImgMSELoss()
    elif loss_type in ["pred_slot_mse"]:
        loss = PredSlotMSELoss()
    return loss


class MSELoss(nn.Module):
    """
    Overriding MSE Loss
    """

    def __init__(self):
        """
        Module initializer
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Computing loss
        """
        if "pred_imgs" not in kwargs:
            raise ValueError("'pred_imgs' must be given to LossTracker to compute 'MSELoss'")
        if "target_imgs" not in kwargs:
            raise ValueError("'target_imgs' must be given to LossTracker to compute 'MSELoss'")
        preds, targets = kwargs.get("pred_imgs"), kwargs.get("target_imgs")
        loss = self.mse(preds, targets)
        return loss


class PredImgMSELoss(nn.Module):
    """
    Pretty much the same MSE Loss.
    Use this loss on predicted images, while still enforcing MSELoss on predicted slots
    """

    def __init__(self):
        """
        Module initializer
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Computing loss
        """
        if "pred_imgs" not in kwargs:
            raise ValueError("'pred_imgs' must be given to LossTracker to compute 'PredImgMSELoss'")
        if "target_imgs" not in kwargs:
            raise ValueError("'target_imgs' must be given to LossTracker to compute 'PredImgMSELoss'")
        preds, targets = kwargs.get("pred_imgs"), kwargs.get("target_imgs")
        loss = self.mse(preds, targets)
        return loss


class PredSlotMSELoss(nn.Module):
    """
    MSE Loss used on slot-like representations. This can be used when forecasting future slots.
    """

    def __init__(self):
        """
        Module initializer
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        """
        Computing loss
        """
        if "preds" not in kwargs:
            raise ValueError("'pred' must be given to LossTracker to compute 'PredSlotMSELoss'")
        if "targets" not in kwargs:
            raise ValueError("'target_slots' must be given to LossTracker to compute 'PredSlotMSELoss'")
        preds, targets = kwargs.get("preds"), kwargs.get("targets")
        loss = self.mse(preds, targets)
        return loss

#
