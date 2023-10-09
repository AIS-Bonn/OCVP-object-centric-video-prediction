"""
Global configurations
"""

import os

CONFIG = {
    "random_seed": 14,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "num_workers": 8,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
    }
}


# Supported datasets, models, metrics, and so on
DATASETS = ["OBJ3D", "MoviA", "MoviC", "ReachSpecificTarget_0Distractors_LargeTargets_DenseReward_DictstateObs_FrontviewViewpointSimplifiedRobot-dataset"]
LOSSES = [
        "mse", "l2",      # standard losses
        "pred_img_mse",   # ||pred_img - target_img||^2 when predicting future images
        "pred_slot_mse",  # ||pred_slot - target_slot||^2 when (also) predicting future slots
    ]
METRICS = [
        "segmentation_ari", "IoU",          # object-centric metrics
        "mse", "psnr", "ssim", "lpips"      # video predicition metrics
    ]
MODELS = ["SAVi"]
PREDICTORS = ["LSTM", "Transformer", "OCVP-Seq", "OCVP-Par", "CondTransformer"]
INITIALIZERS = ["Random", "LearnedRandom", "Learned", "Masks", "CoM", "BBox"]

DEFAULTS = {
    "dataset": {
        "dataset_name": "OBJ3D",
        "shuffle_train": True,
        "shuffle_eval": False,
        "use_segmentation": True,
        "target": "rgb",
        "random_start": True
    },
    "model": {
        "model_name": "SAVi",
        "SAVi": {
            "num_slots": 6,
            "slot_dim": 64,
            "in_channels": 3,
            "encoder_type": "ConvEncoder",
            "num_channels": (32, 32, 32, 32),
            "mlp_encoder_dim": 64,
            "mlp_hidden": 128,
            "num_channels_decoder": (32, 32, 32, 32),
            "kernel_size": 5,
            "num_iterations_first": 3,
            "num_iterations": 1,
            "resolution": (64, 64),
            "downsample_encoder": False,
            "downsample_decoder": False,
            "decoder_resolution": (64, 64),  # set it according to size after downsampling if necessary
            "upsample": 2,
            "use_predictor": True,
            "initializer": "LearnedRandom"
        },
        "predictor":
            {
            "predictor_name": "Transformer",
            "LSTM": {
                "num_cells": 2,
                "hidden_dim": 64,
                "residual": True,
            },
            "Transformer": {
                "token_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "n_heads": 4,
                "residual": True,
                "input_buffer_size": None
            },
            "OCVP-Seq": {
                "token_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "n_heads": 4,
                "residual": True,
                "input_buffer_size": None
            },
            "OCVP-Par": {
                "token_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "n_heads": 4,
                "residual": True,
                "input_buffer_size": None
            },
            "CondTransformer": {
                "cond_dim": 4,
                "token_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "n_heads": 4,
                "residual": True,
                "input_buffer_size": None
            }
        }
    },
    "loss": [
        {
            "type": "mse",
            "weight": 1
        }
    ],
    "predictor_loss": [
        {
          "type": "pred_img_mse",
          "weight": 1
        },
        {
          "type": "pred_slot_mse",
          "weight": 1
        }
    ],
    "training_slots": {  # training related parameters
        "num_epochs": 1000,        # number of epochs to train for
        "save_frequency": 10,   # saving a checkpoint after these iterations ()
        "log_frequency": 25,     # logging stats after this amount of updates
        "image_log_frequency": 100,     # logging stats after this amount of updates
        "batch_size": 64,
        "lr": 1e-4,
        "optimizer": "adam",      # optimizer parameters: name, L2-reg, momentum
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "",             # learning rate scheduler parameters
        "lr_factor": 0.8,            # Meaning depends on scheduler. See lib/model_setup.py
        "patience": 10,
        "scheduler_steps": 1e6,
        "lr_warmup": False,       # learning rate warmup parameters (2 epochs or 200 iters default)
        "warmup_steps": 2000,
        "warmup_epochs": 2,
        "gradient_clipping": True,
        "clipping_max_value": 0.05  # according to SAVI paper
    },
    "training_prediction": {  # training related parameters
        "num_context": 5,
        "num_preds": 5,
        "teacher_force": False,
        "skip_first_slot": False,  # if True, slots from first image are not considered
        "num_epochs": 1500,        # number of epochs to train for
        "train_iters_per_epoch": 1e10,   # max number of iterations per epoch
        "save_frequency": 10,   # saving a checkpoint after these iterations ()
        "save_frequency_iters": 1000,   # saving a checkpoint after these iterations
        "log_frequency": 25,     # logging stats after this amount of updates
        "image_log_frequency": 100,     # logging stats after this amount of updates
        "batch_size": 64,
        "sample_length": 10,
        "gradient_clipping": True,
        "clipping_max_value": 0.05  # according to SAVI paper
    }
}


COLORS = ["white", "blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "darkviolet", "springgreen",
          "aqua", "royalblue", "navy", "forestgreen", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]


#
