"""
Modules for the initalization of the slots on SlotAttention and SAVI
"""

import torch
import torch.nn as nn
from math import sqrt

from CONFIG import INITIALIZERS


ENCODER_RESOLUTION = (8, 14)


def get_initalizer(mode, slot_dim, num_slots, encoder_resolution=None):
    """
    Fetching the initializer module of the slots

    Args:
    -----
    model: string
        Type of initializer to use. Valid modes are {INITIALIZERS}
    slot_dim: int
        Dimensionality of the slots
    num_slots: int
        Number of slots to initialize
    """
    encoder_resolution = encoder_resolution if encoder_resolution is not None else ENCODER_RESOLUTION
    if mode not in INITIALIZERS:
        raise ValueError(f"Unknown initializer {mode = }. Available modes are {INITIALIZERS}")

    if mode == "Random":
        intializer = Random(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "LearnedRandom":
        intializer = LearnedRandom(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "Learned":
        intializer = Learned(slot_dim=slot_dim, num_slots=num_slots)
    elif mode == "Masks":
        raise NotImplementedError("'Masks' initialization is not supported...")
    elif mode == "CoM":
        intializer = CoordInit(slot_dim=slot_dim, num_slots=num_slots, mode="CoM")
    elif mode == "BBox":
        intializer = CoordInit(slot_dim=slot_dim, num_slots=num_slots, mode="BBox")
    else:
        raise ValueError(f"UPSI, {mode = } should not have reached here...")

    return intializer


class Random(nn.Module):
    """
    Gaussian random slot initialization
    """

    def __init__(self, slot_dim, num_slots):
        """
        Module intializer
        """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

    def forward(self, batch_size, **kwargs):
        """
        Sampling random Gaussian slots
        """
        slots = torch.randn(batch_size, self.num_slots, self.slot_dim)
        return slots


class LearnedRandom(nn.Module):
    """
    Learned random intialization. This is the default mode used in SlotAttention.
    Slots are randomly sampled from a Gaussian distribution. However, the statistics of this
    distribution (mean vector and diagonal of covariance) are learned via backpropagation
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_sigma, -limit, limit)
        return

    def forward(self, batch_size, **kwargs):
        """
        Sampling random slots from the learned gaussian distribution
        """
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_sigma.expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=self.slots_mu.device)
        return slots


class Learned(nn.Module):
    """
    Learned intialization.
    For each slot a discrete initialization is learned via backpropagation.
    """

    def __init__(self, slot_dim, num_slots):
        """ Module intializer """
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots

        self.initial_slots = torch.nn.ParameterList(torch.nn.Parameter(torch.randn(1, 1, self.slot_dim)) for _ in range(self.num_slots))

        with torch.no_grad():
            limit = sqrt(6.0 / (1 + slot_dim))
            for i in range(num_slots):
                torch.nn.init.uniform_(self.initial_slots[i], -limit, limit)
        return

    def forward(self, batch_size, **kwargs):
        """
        Return learned slot initializations
        """
        slot_list = [self.initial_slots[i].expand(batch_size, 1, -1) for i in range(self.num_slots)]
        return torch.cat(slot_list, dim=1)


class CoordInit(nn.Module):
    """
    Slots are initalized by encoding, for each object, the coordinates of one of the following:
        - the CoM of the instance segmentation of each object, represented as [y, x]
        - the BBox containing each object, represented as [y_min, x_min, y_max, x_max]
    """

    MODES = ["CoM", "BBox"]
    MODE_REP = {
            "CoM": "com_coords",
            "BBox": "bbox_coords"
        }
    IN_FEATS = {
            "CoM": 2,
            "BBox": 4
        }

    def __init__(self, slot_dim, num_slots, mode):
        """
        Module intializer
        """
        assert mode in CoordInit.MODES, f"Unknown {mode = }. Use one of {CoordInit.MODES}"
        super().__init__()
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.mode = mode
        self.coord_encoder = nn.Sequential(
                nn.Linear(CoordInit.IN_FEATS[self.mode], 256),
                nn.ReLU(),
                nn.Linear(256, slot_dim),
            )
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))
        return

    def forward(self, batch_size, **kwargs):
        """
        Encoding BBox or CoM coordinates into slots using an MLP
        """
        device = self.dummy_parameter.device
        rep_name = CoordInit.MODE_REP[self.mode]
        in_feats = CoordInit.IN_FEATS[self.mode]

        coords = kwargs.get(rep_name, None)
        if coords is None or coords.sum() == 0:
            raise ValueError(f"{self.mode} Initializer requires having '{rep_name}'...")
        if len(coords.shape) == 4:  # getting only coords corresponding to time-step t=0
            coords = coords[:, 0]
        coords = coords.to(device)

        # obtaining -1-vectors for filling the slots that currently do not have an object
        num_coords = coords.shape[1]
        if num_coords > self.num_slots:
            raise ValueError(f"There shouldnt be more {num_coords = } than {self.num_slots = }! ")
        if num_coords < self.num_slots:
            remaining_masks = self.num_slots - num_coords
            pad_zeros = -1 * torch.ones((coords.shape[0], remaining_masks, in_feats), device=device)
            coords = torch.cat([coords, pad_zeros], dim=2)

        slots = self.coord_encoder(coords)
        return slots


#
