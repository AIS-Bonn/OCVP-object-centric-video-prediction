"""
Methods for loading specific datasets, fitting data loaders and other
"""

# from torchvision import datasets
from torch.utils.data import DataLoader
from data import OBJ3D, MOVI, RobotDataset
from CONFIG import CONFIG, DATASETS


def load_data(exp_params, split="train"):
    """
    Loading a dataset given the parameters

    Args:
    -----
    dataset_name: string
        name of the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    """
    dataset_name = exp_params["dataset"]["dataset_name"]

    if dataset_name == "OBJ3D":
        dataset = OBJ3D(
                mode=split,
                sample_length=exp_params["training_prediction"]["sample_length"]
            )
    elif dataset_name == "MoviA":
        dataset = MOVI(
                datapath="/home/nfs/inf6/data/datasets/MOVi/movi_a",
                target=exp_params["dataset"].get("target", "rgb"),
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                random_start=exp_params["dataset"].get("random_start", False),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "MoviC":
        dataset = MOVI(
                datapath="/home/nfs/inf6/data/datasets/MOVi/movi_c",
                target=exp_params["dataset"].get("target", "rgb"),
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                random_start=exp_params["dataset"].get("random_start", False),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif "Robot-dataset" in dataset_name:
        dataset = RobotDataset(
                mode=split,
                dataset_name=dataset_name,
                sample_length=exp_params["training_prediction"]["sample_length"]
            )
    else:
        raise NotImplementedError(
                f"""ERROR! Dataset'{dataset_name}' is not available.
                Please use one of the following: {DATASETS}..."""
            )

    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )

    return data_loader


def unwrap_batch_data(exp_params, batch_data):
    """
    Unwrapping the batch data depending on the dataset that we are training on
    """
    initializer_kwargs = {}
    condition = None
    if exp_params["dataset"]["dataset_name"] in ["OBJ3D"]:
        videos, targets, _ = batch_data
    elif exp_params["dataset"]["dataset_name"] in ["MoviA", "MoviC"]:
        videos, targets, all_reps = batch_data
        initializer_kwargs["instance_masks"] = all_reps["masks"]
        initializer_kwargs["com_coords"] = all_reps["com_coords"]
        initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
    elif "Robot-dataset" in exp_params["dataset"]["dataset_name"]:
      videos, targets, condition, _ = batch_data
    else:
        dataset_name = exp_params["dataset"]["dataset_name"]
        raise NotImplementedError(f"Dataset {dataset_name} is not supported...")
    return videos, targets, condition, initializer_kwargs


def unwrap_batch_data_masks(exp_params, batch_data):
    """
    Unwrapping the batch data for a mask-based evaluation depending on the dataset that
    we are currently evaluating on
    """
    dbs = ["MoviA", "MoviC"]
    dataset_name = exp_params["dataset"]["dataset_name"]
    initializer_kwargs = {}
    condition = None
    if dataset_name in ["MoviA", "MoviC"]:
        videos, _, all_reps = batch_data
        masks = all_reps["masks"]
        initializer_kwargs["instance_masks"] = all_reps["masks"]
        initializer_kwargs["com_coords"] = all_reps["com_coords"]
        initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
    else:
        raise ValueError(f"Only {dbs} support object-based mask evaluation. Given {dataset_name = }")
    return videos, masks, condition, initializer_kwargs


#
