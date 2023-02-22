"""
Dataclass and loading of the MOVI dataset from the Tensorflow files.

https://github.com/google-research/kubric/tree/main/challenges/movi
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tensorflow_datasets as tfds
# from itertools import islice
import tensorflow as tf
import lib.visualizations as visualizations

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


class _MOVI(Dataset):
    """
    DataClass for the MOVI dataset.

    Args:
    -----
    movi_type: string
        Type of MOVI dataset to use
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    img_size: tuple
        Images are resized to this resolution
    """

    MAX_OBJS = 11

    def __init__(self, split, num_frames, img_size=(64, 64), slot_initializer="LearnedInit"):
        """ Dataset initializer """
        assert split in ["train", "val", "valid", "validation", "test"]
        split = "validation" if split in ["val", "valid", "validation"] else split

        # dataset parameters
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.slot_initializer = slot_initializer
        self.get_masks = False
        self.get_bbox = False

        # resizer modules for the images and masks respectively
        self.resizer = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        self.resizer_mask = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        # loading data
        self.db, self.len_db = self._load_data()
        return

    def __len__(self):
        """ Number of sequences in dataset """
        return self.len_db

    def __getitem__(self, i):
        """
        Sampling a sequence from the dataset
        """
        all_data = next(self.db)
        # all_data = next(islice(self.db, i, i+1))
        # all_data = self.db[i]

        # images
        imgs = torch.from_numpy(all_data["video"])[:self.num_frames].permute(0, 3, 1, 2) / 255
        imgs = self.resizer(imgs).float()

        # instance segmentations
        segmentation = torch.from_numpy(all_data["segmentations"][:self.num_frames, ..., 0])
        segmentation = self.resizer_mask(segmentation)

        # coordinates
        bbox, com = self._get_bbox_com(all_data, imgs)

        # optical flow
        minv, maxv = all_data["metadata"]["forward_flow_range"]
        forward_flow = all_data["forward_flow"] / 65535 * (maxv - minv) + minv
        flow_rgb = visualizations.flow_to_rgb(forward_flow)
        flow_rgb = torch.from_numpy(flow_rgb).permute(0, 3, 1, 2)
        flow_rgb = self.resizer_mask(flow_rgb)

        data = {
                "frames": imgs,
                "masks": segmentation,
                "com_coords": com,
                "bbox_coords": bbox,
                "flow": flow_rgb
            }
        return imgs, data

    def _get_bbox_com(self, all_data, imgs):
        """
        Obtaining BBox information
        """
        bboxes = all_data["instances"]["bboxes"].numpy()
        bbox_frames = all_data["instances"]["bbox_frames"].numpy()
        num_frames, _, H, W = imgs.shape
        num_objects = bboxes.shape[0]
        com = torch.zeros(num_frames, num_objects, 2)
        bbox = torch.zeros(num_frames, num_objects, 4)
        for t in range(num_frames):
            for k in range(num_objects):
                if t in bbox_frames[k]:
                    idx = np.nonzero(bbox_frames[k] == t)[0][0]
                    min_y, min_x, max_y, max_x = bboxes[k][idx]
                    min_y, min_x = max(1, min_y * H), max(1, min_x * W)
                    max_y, max_x = min(H - 1, max_y * H), min(W - 1, max_x * W)
                    bbox[t, k] = torch.tensor([min_x, min_y, max_x, max_y])
                    com[t, k] = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2]).round()
                else:
                    bbox[t, k] = torch.ones(4) * -1
                    com[t, k] = torch.ones(2) * -1

        # padding so as to batch BBoxes or CoMs
        if num_objects < self.MAX_OBJS:
            rest = self.MAX_OBJS - num_objects
            rest_bbox = torch.ones((bbox.shape[0], rest, 4), device=imgs.device) * -1
            rest_com = torch.ones((bbox.shape[0], rest, 2), device=imgs.device) * -1
            bbox = torch.cat([bbox, rest_bbox], dim=1)
            com = torch.cat([com, rest_com], dim=1)

        return bbox, com


class _MoviA(_MOVI):
    """
    DataClass for the Movi-A dataset.
    It contains CLEVR-like objects on a grey environment. Objects collide with each other
    """

    def _load_data(self):
        """ Loading MOVI-A data"""
        print(f"Loading MOVI-A {self.split} set...")
        dataset_builder = tfds.builder(
                "movi_a/128x128:1.0.0",
                data_dir="/home/nfs/inf6/data/datasets/MOVi"
            )
        split = "train" if self.split == "train" else "validation"
        ds = tfds.as_numpy(dataset_builder.as_dataset(split=split))
        len_ds = len(ds)

        new_ds = iter(ds)
        return new_ds, len_ds


class _MoviC(_MOVI):
    """
    DataClass for the Movi-A dataset.
    Complex objects on a grey environment. Objects collide with each other
    """

    def _load_data(self):
        """ Loading MOVI-B data"""
        print(f"Loading MOVI-C {self.split} set...")
        dataset_builder = tfds.builder(
                "movi_c/128x128:1.0.0",
                data_dir="/home/nfs/inf6/data/datasets/MOVi"
            )
        split = "train" if self.split == "train" else "validation"
        ds = tfds.as_numpy(dataset_builder.as_dataset(split=split))
        len_ds = len(ds)

        new_ds = iter(ds)
        return new_ds, len_ds


#
