"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import torch

from lib.logger import print_, log_function
from lib.schedulers import LRWarmUp, ExponentialLRSchedule
from lib.utils import create_directory
import models
import models.Predictors as predictors
from CONFIG import MODELS, PREDICTORS


@log_function
def setup_model(model_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_name = model_params["model_name"]
    if model_name not in MODELS:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models; {MODELS}")
    cur_model_params = model_params[model_name]

    if(model_name == "SAVi"):
        model = models.SAVi(**cur_model_params)
    else:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models; {MODELS}")

    return model


@log_function
def setup_predictor(exp_params):
    """
    Loading the predictor given the predictor parameters stated in the exp_params file

    Args:
    -----
    predictor_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    predictor: PredictorWrapper
        Instanciated predictor given the parameters, wrapped into a PredictorWrapper to
        forecast slots for future time steps.
    """
    # model params
    model_params = exp_params["model"]
    model_name = model_params["model_name"]
    if model_name not in MODELS:
        raise NotImplementedError(f"Model '{model_name}' not in recognized models: {MODELS}")
    cur_model_params = model_params[model_name]

    # predictor params
    predictor_params = model_params["predictor"]
    predictor_name = predictor_params["predictor_name"]
    train_pred_params = exp_params["training_prediction"]
    if predictor_name not in PREDICTORS:
        raise NotImplementedError(f"Predictor '{predictor_name}' not in recognized predictors: {PREDICTORS}")
    cur_predictor_params = predictor_params[predictor_name]

    # instanciating predictor
    if(predictor_name == "LSTM"):
        predictor = predictors.LSTMPredictor(
                slot_dim=cur_model_params["slot_dim"],
                hidden_dim=cur_predictor_params["hidden_dim"],
                num_layers=cur_predictor_params["num_cells"],
                residual=cur_predictor_params.get("residual", True)
            )
    elif(predictor_name == "Transformer"):
        num_seed = train_pred_params["num_context"]
        predictor = predictors.VanillaTransformerPredictor(
                num_slots=cur_model_params["num_slots"],
                slot_dim=cur_model_params["slot_dim"],
                num_imgs=train_pred_params["sample_length"],
                token_dim=cur_predictor_params["token_dim"],
                hidden_dim=cur_predictor_params["hidden_dim"],
                num_layers=cur_predictor_params["num_layers"],
                n_heads=cur_predictor_params["n_heads"],
                residual=cur_predictor_params.get("residual", False),
                input_buffer_size=cur_predictor_params.get("input_buffer_size", num_seed)
            )
    elif(predictor_name == "OCVP-Seq"):
        num_seed = train_pred_params["num_context"]
        predictor = predictors.OCVPSeq(
                num_slots=cur_model_params["num_slots"],
                slot_dim=cur_model_params["slot_dim"],
                num_imgs=train_pred_params["sample_length"],
                token_dim=cur_predictor_params["token_dim"],
                hidden_dim=cur_predictor_params["hidden_dim"],
                num_layers=cur_predictor_params["num_layers"],
                n_heads=cur_predictor_params["n_heads"],
                residual=cur_predictor_params.get("residual", False),
                input_buffer_size=cur_predictor_params.get("input_buffer_size", num_seed)
            )
    elif(predictor_name == "OCVP-Par"):
        num_seed = train_pred_params["num_context"]
        predictor = predictors.OCVPPar(
                num_slots=cur_model_params["num_slots"],
                slot_dim=cur_model_params["slot_dim"],
                num_imgs=train_pred_params["sample_length"],
                token_dim=cur_predictor_params["token_dim"],
                hidden_dim=cur_predictor_params["hidden_dim"],
                num_layers=cur_predictor_params["num_layers"],
                n_heads=cur_predictor_params["n_heads"],
                residual=cur_predictor_params.get("residual", False),
                input_buffer_size=cur_predictor_params.get("input_buffer_size", num_seed)
            )
    elif(predictor_name == "CondTransformer"):
        num_seed = train_pred_params["num_context"]
        predictor = predictors.CondTransformerPredictor(
                num_slots=cur_model_params["num_slots"],
                slot_dim=cur_model_params["slot_dim"],
                num_imgs=train_pred_params["sample_length"],
                token_dim=cur_predictor_params["token_dim"],
                hidden_dim=cur_predictor_params["hidden_dim"],
                cond_dim=cur_predictor_params["cond_dim"],
                num_layers=cur_predictor_params["num_layers"],
                n_heads=cur_predictor_params["n_heads"],
                residual=cur_predictor_params.get("residual", False),
                input_buffer_size=cur_predictor_params.get("input_buffer_size", num_seed)
            )
    else:
        raise NotImplementedError(f"Predictor '{predictor_name}' not in recognized predictors: {PREDICTORS}")

    # instanciating predictor wrapper module to iterate over the data
    predictor = predictors.PredictorWrapper(
            exp_params=exp_params,
            predictor=predictor
        )
    return predictor


@log_function
def save_checkpoint(model, optimizer, scheduler, lr_warmup, epoch, exp_path,
                    finished=False, savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    scheduler: Object
        Learning rate scheduler to save
    lr_warmup: Object
        Module performing learning rate warm-up
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data,
            "lr_warmup": lr_warmup
        }, savepath)

    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    if(checkpoint_path is None):
        return model

    # loading model to either cpu or cpu
    if(map_cpu):
        checkpoint = torch.load(checkpoint_path,  map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)

    # wrapping predictor into PredictorWrapper for backwards compatibility
    first_key_model = list(model.state_dict().keys())[0]
    first_key_checkpoint = list(checkpoint['model_state_dict'].keys())[0]
    if first_key_model.startswith("predictor") and not first_key_checkpoint.startswith("predictor"):
        checkpoint['model_state_dict'] = {
                f"predictor.{key}": val for key, val in checkpoint['model_state_dict'].items()
            }

    # loading model parameters.
    model.load_state_dict(checkpoint['model_state_dict'])

    # returning only model for transfer learning or returning also optimizer for resuming training
    if(only_model):
        return model

    # returning all other necessary objects
    optimizer, scheduler, lr_warmup = kwargs["optimizer"], kwargs["scheduler"], kwargs["lr_warmup"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if "lr_warmup" in checkpoint:
        lr_warmup.load_state_dict(checkpoint['lr_warmup'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, lr_warmup, epoch


@log_function
def setup_optimizer(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """

    lr = exp_params["training_slots"]["lr"]
    lr_factor = exp_params["training_slots"]["lr_factor"]
    patience = exp_params["training_slots"]["patience"]
    momentum = exp_params["training_slots"]["momentum"]
    optimizer = exp_params["training_slots"]["optimizer"]
    nesterov = exp_params["training_slots"]["nesterov"]
    scheduler = exp_params["training_slots"]["scheduler"]
    scheduler_steps = exp_params["training_slots"].get("scheduler_steps", 1e6)

    # SGD-based optimizer
    if(optimizer == "adam"):
        print_("Setting up Adam optimizer:")
        print_(f"    LR: {lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        print_("Setting up SGD optimizer:")
        print_(f"    LR: {lr}")
        print_(f"    Momentum: {momentum}")
        print_(f"    Nesterov: {nesterov}")
        print_("    Weight Decay: 0.0005")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)

    # LR-scheduler
    if (scheduler == "constant"):
        print_("Setting up Constant LR-Scheduler:")
        print_(f"   Factor:   {lr_factor}")
        print_(f"   total_iters: {scheduler_steps}")
        scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=1,
                total_iters=scheduler_steps
            )
    elif(scheduler == "plateau"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"   Patience: {patience}")
        print_(f"   Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif(scheduler == "step"):
        print_("Setting up Step LR-Scheduler")
        print_(f"   Step Size: {patience}")
        print_(f"   Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )

    elif(scheduler == "multi_step"):
        print_("Setting up MultiStepLR LR-Scheduler")
        print_(f"   Milestones: {patience}")
        print_(f"   Factor:    {lr_factor}")
        if not isinstance(patience, list):
            raise ValueError(f"Milestones ({patience}) must be a list of increasing integers...")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                milestones=patience
            )

    elif(scheduler == "exponential"):
        print_("Setting up Exponential LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   Factor:  {lr_factor}")
        print_(f"   Steps:   {scheduler_steps}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor,
                total_steps=scheduler_steps
            )
    elif(scheduler == "cosine_annealing"):
        print_("Setting up Cosine Annealing LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   Factor:  {lr_factor}")
        print_(f"   T_max:   {scheduler_steps}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=scheduler_steps
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    # seting up lr_warmup object
    lr_warmup = setup_lr_warmup(params=exp_params["training_slots"])

    return optimizer, scheduler, lr_warmup


@log_function
def setup_lr_warmup(params):
    """
    Seting up the learning rate warmup handler given experiment params

    Args:
    -----
    params: dictionary
        training parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    lr_warmup: Object
        object that steadily increases the learning rate during the first iterations.

    Example:
    -------
        #  Learning rate is initialized with 3e-4 * (1/1000). For the first 1000 iterations
        #  or first epoch, the learning rate is updated to 3e-4 * (iter/1000).
        # after the warmup period, learning rate is fixed at 3e-4
        optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
        lr_warmup = LRWarmUp(init_lr=3e-4, warmup_steps=1000, max_epochs=1)
        ...
        lr_warmup(iter=cur_iter, epoch=cur_epoch, optimizer=optimizer)  # updating lr
    """
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if(use_warmup):
        warmup_steps = params["warmup_steps"]
        warmup_epochs = params["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
        print_("Setting up learning rate warmup:")
        print_(f"  Target LR:     {lr}")
        print_(f"  Warmup Steps:  {warmup_steps}")
        print_(f"  Warmup Epochs: {warmup_epochs}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup


def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):

    Note: this does not work for predictors, since it saves the backbone, but not the
          predictor model.
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.warmup_scheduler.scheduler,
                    lr_warmup=self_.warmup_scheduler.lr_warmup,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except


#
