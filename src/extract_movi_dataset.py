"""
Converting the MOVi dataset from the shitty tensorboard format into images, so that we can
then load them like civilised people without a million Tensflow errorsand warnings
"""

import os
import torch
import torchvision
from tqdm import tqdm
from data.MoviConvert import _MoviA
import lib.utils as utils
from CONFIG import CONFIG


PATH = os.path.join(CONFIG["paths"]["data_path"], "movi_a")


def process_dataset(split="train"):
    """ """
    data_path = os.path.join(PATH, split)
    utils.create_directory(data_path)

    db = _MoviA(split=split, num_frames=100, img_size=(128, 128))
    db.get_masks = True
    db.get_bbox = True
    print(f"  --> {len(db) = }")

    # iterating and saving data
    for i in tqdm(range(len(db))):
        imgs, all_preds = db[i]
        bbox = all_preds["bbox_coords"]
        com = all_preds["com_coords"]
        masks = all_preds["masks"]
        flow = all_preds["flow"]
        for j in range(imgs.shape[0]):
            torchvision.utils.save_image(flow[j], fp=os.path.join(data_path, f"flow_{i:05d}_{j:02d}.png"))
            torchvision.utils.save_image(imgs[j], fp=os.path.join(data_path, f"rgb_{i:05d}_{j:02d}.png"))
        torch.save({"com": com, "bbox": bbox}, os.path.join(data_path, f"coords_{i:05d}.pt"))
        torch.save({"masks": masks}, os.path.join(data_path, f"mask_{i:05d}.pt"))

    return


if __name__ == '__main__':
    os.system("clear")
    print("Processing Validation set")
    process_dataset(split="validation")
    print("Processing Training set")
    process_dataset(split="train")
    print("Finished")
