"""
DataClass and loading any version of MOVi dataset (e.g. MOVi-A or MOVi-C)

This datasets assumes that the original MOVi file have been extracted from TFRecords and
preprocessed into images.
Code to extract and preprocess the MOVi TFRecords can be found in TODO

 - Source: https://github.com/google-research/kubric/tree/main/challenges/movi
"""

import os
import imageio
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]


class MOVI(Dataset):
    """
    DataClass for the MOVI dataset.

    Args:
    -----
    datapath: string
        Path to the directory where the current MOVi data is stored.
        Directory must be named as the MOVi version. For instance, 'movi_a' or 'movi_c'.
    target: string
        Datatype of the target data to load. It can be 'rgb' or 'flow'.
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load.
    img_size: tuple
        Images are resized to this resolution.
    random_start: bool
        If True, first frame of the sequence is sampled at random between the possible starting frames.
        Otherwise, starting frame is always the first frame in the sequence.
    slot_initializer: string
        Initialization mode used to initialize the slots
    """

    DATA_TYPE = ["movi_a", "movi_c"]
    TARGETS = ["rgb", "flow"]
    MAX_OBJS = 11
    NUM_FRAMES = 24

    def __init__(self, datapath, target="rgb", split="validation", num_frames=24, img_size=(64, 64),
                 random_start=False, slot_initializer="LearnedInit"):
        """
        Dataset initializer
        """
        assert target in MOVI.TARGETS, f"Unknow {target = }. Use one of {MOVI.TARGETS}..."
        assert split in ["train", "val", "valid", "validation", "test"], f"Unknown {split}..."
        split = "validation" if split in ["val", "valid", "validation", "test"] else split

        data_type = datapath.split("/")[-1]
        assert data_type in MOVI.DATA_TYPE, f"Unknown {data_type = }. Use one of {MOVI.DATA_TYPE}"

        # dataset parameters
        self.cur_data_path = os.path.join(datapath, split)
        if not os.path.exists(self.cur_data_path):
            raise FileNotFoundError(f"Data path {self.cur_data_path} does not exist")
        self.target = target
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.random_start = random_start
        self.slot_initializer = slot_initializer
        self.get_rgb = True if target == "rgb" else False
        self.get_flow = True if target == "flow" else False
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
        self.db = self._load_data()
        return

    def __len__(self):
        """
        Number of sequences in dataset
        """
        return len(self.db)

    def __getitem__(self, i):
        """
        Sampling a sequence from the dataset
        """
        all_data = self.db[i]
        if self.random_start and self.split == "train":
            start_frame = np.random.randint(0, MOVI.NUM_FRAMES - self.num_frames)
        else:
            start_frame = 0

        # Loading images. They are always the input to the model, and mostly the are also the target
        imgs = []
        img_paths = all_data["imgs"][start_frame:start_frame+self.num_frames]
        imgs = [imageio.imread(frame) / 255. for frame in img_paths]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        imgs = self.resizer(imgs).float()

        # loading optical flow if necessary, which can be used as target.
        flow = []
        if self.get_flow:
            flow_paths = all_data["flow"][start_frame:start_frame+self.num_frames]
            flow = [imageio.imread(flow) for flow in flow_paths]
            flow = np.stack(flow, axis=0)
            flow = torch.from_numpy(flow).permute(0, 3, 1, 2) / 255.
            flow = self.resizer_mask(flow).float()

        # default representation is either RGB-frames or optical flow.
        targets = imgs if self.target == "rgb" else flow

        # loading instance segmentation masks if necessary. Can be used as conditioning or for eval
        segmentation = []
        if self.get_masks or self.slot_initializer == "Masks":
            segmentation = torch.load(all_data["masks"])
            segmentation = segmentation["masks"][start_frame:start_frame+self.num_frames]
            segmentation = np.stack(segmentation, axis=0)
            segmentation = torch.from_numpy(segmentation)
            segmentation = self.resizer_mask(segmentation)

        # loading center of mass and bounding box, if necessary. Can be used as conditioning
        com, bbox = [], []
        if self.get_bbox or self.slot_initializer in ["CoM", "BBox"]:
            coords = torch.load(all_data["coords"])
            bbox, com = coords["bbox"], coords["com"]
            bbox = bbox * imgs.shape[-1] / 128
            com = com * imgs.shape[-1] / 128
            bbox = bbox[start_frame:start_frame+self.num_frames]
            com = com[start_frame:start_frame+self.num_frames]

        data = {
                "frames": imgs,
                "flow": flow,
                "masks": segmentation,
                "com_coords": com,
                "bbox_coords": bbox,
            }
        return imgs, targets, data

    def _get_bbox_com(self, all_data, imgs):
        """
        Obtaining BBox and Center of Mass coordinate information from instance segmentation masks.

        BBox and Coords are stacked into torch tensors with shape:
            - BBox: (num_frames, num_objs, 4)
            - CoM:  (num_frames, num_objs, 2)

        If the actualy number of objects is smaller than the expected one, dummy tensors filled with -1 are
        used to fill the coordinates. This alleviates issues when stacking into batches.
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

        # padding so as to stack multiple BBox or CoM coordinate tensors into batches.
        if num_objects < self.MAX_OBJS:
            rest = self.MAX_OBJS - num_objects
            rest_bbox = torch.ones((bbox.shape[0], rest, 4), device=imgs.device) * -1
            rest_com = torch.ones((bbox.shape[0], rest, 2), device=imgs.device) * -1
            bbox = torch.cat([bbox, rest_bbox], dim=1)
            com = torch.cat([com, rest_com], dim=1)

        return bbox, com

    def _load_data(self):
        """
        Loading the data into a nice dictionary with the following structure:
            {seq_num: {imgs: [], flow: [], "masks": [], "coords": []}
        """
        db = {}
        all_files = os.listdir(self.cur_data_path)
        seqs = sorted(list(set([int(f.split("_")[1]) for f in all_files if "rgb" in f])))
        for seq in tqdm(seqs):
            db[seq] = {}
            db[seq]["imgs"] = [
                    os.path.join(self.cur_data_path, f"rgb_{seq:05d}_{i:02d}.png") for i in range(24)
                ]
            db[seq]["flow"] = [
                    os.path.join(self.cur_data_path, f"flow_{seq:05d}_{i:02d}.png") for i in range(24)
                ]
            db[seq]["masks"] = os.path.join(self.cur_data_path, f"mask_{seq:05d}.pt")
            db[seq]["coords"] = os.path.join(self.cur_data_path, f"coords_{seq:05d}.pt")
        return db


#
