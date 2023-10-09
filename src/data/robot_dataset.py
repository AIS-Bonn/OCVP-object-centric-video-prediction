import numpy as np
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image

from data import OBJ3D
from CONFIG import CONFIG

PATH = CONFIG["paths"]["data_path"]


class RobotDataset(OBJ3D):
    """
    DataClass for datasets generated with the visual-block-builder package.

    During training, we sample a random subset of frames in the episode. At inference time,
    we always start from the first frame, e.g., when the ball moves towards the objects, and
    before any collision happens.

    Args:
    -----
    mode: string
        Dataset split to load. Can be one of ['train', 'val', 'test']
    ep_len: int
        Number of frames in an episode. Default is 30
    sample_length: int
        Number of frames in the sequences to load
    random_start: bool
        If True, first frame of the sequence is sampled at random between the possible starting frames.
        Otherwise, starting frame is always the first frame in the sequence.
    """

    def __init__(self, mode, dataset_name, ep_len=30, sample_length=20, random_start=True):
        """
        Dataset Initializer
        """
        super().__init__(mode, ep_len, sample_length, random_start)

        assert mode in ["train", "val", "valid", "eval", "test"], f"Unknown dataset split {mode}..."
        mode = "val" if mode in ["val", "valid"] else mode
        mode = "test" if mode in ["test", "eval"] else mode
        assert mode in ['train', 'val', 'test'], f"Unknown dataset split {mode}..."

        self.root = os.path.join(PATH, dataset_name, mode)
        self.mode = mode
        self.sample_length = sample_length
        self.random_start = random_start

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        # episode-related paramters
        self.epsisodes = []
        self.EP_LEN = ep_len
        if mode == "train" and self.random_start:
            self.seq_per_episode = self.EP_LEN - self.sample_length + 1
        else:
            self.seq_per_episode = 1

        # loading images from data directories and assembling then into episodes
        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, '*.png')))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)
        return

    def __getitem__(self, index):
        """
        Fetching a sequence from the dataset
        """
        imgs = []

        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.epsisodes[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((64, 64))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)
        img = torch.stack(imgs, dim=0).float()

        # load actions
        actions = torch.from_numpy(np.load("/" + osp.join(*(self.epsisodes[ep][0].split("/")[:-1] + ["actions.npy"])))[offset:end])

        targets = img
        all_reps = {"videos": img}
        return img, targets, actions, all_reps
