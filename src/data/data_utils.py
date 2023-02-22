"""
Some data processing and other utils
"""

import numpy as np
import torch


def get_slots_stats(seq, masks):
    """
    Obtaining stats about the number of slots in a video sequence

    Args:
    -----
    seq: torch tensor
        Sequence of images. Shape is (N_frames, N_channels, H, W)
    masks: torch Tensor
        Instance segmentation masks. Shape is (N_frames, 1, H, W)
    """
    total_num_slots = len(torch.unique(masks))
    slot_dist = [len(torch.unique(m)) for m in masks]

    stats = {
            "total_num_slots": total_num_slots,
            "slot_dist": slot_dist,
            "max_num_slots": np.max(slot_dist),
            "min_num_slots": np.min(slot_dist)
        }
    return stats


def masks_to_boxes(masks):
    """
    Converting a binary segmentation mask into a bounding box

    Args:
    -----
    masks: torch Tensor
        Segmentation masks. Shape is (n_imgs, n_objs, H, W)

    Returns:
    --------
    bboxes: torch Tensor
        Bounding boxes corresponding the input segmentation masks in format [x1, y1, x2, y2].
        Shape is (n_imgs, 4)
    """
    assert masks.unique().tolist() == [0, 1]

    bboxes = torch.zeros(masks.shape[0], 4)
    for i, mask in enumerate(masks):
        if mask.max() == 0:
            bboxes[i] = torch.ones(4) * -1
            continue
        vertical_indices = torch.where(torch.any(mask, axis=1))[0]
        horizontal_indices = torch.where(torch.any(mask, axis=0))[0]
        if horizontal_indices.shape[0]:
            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]
        else:
            bboxes[i] = torch.ones(4) * -1
            continue
        bboxes[i] = torch.tensor([x1, y1, x2, y2])
    return bboxes.to(masks.device)

#
