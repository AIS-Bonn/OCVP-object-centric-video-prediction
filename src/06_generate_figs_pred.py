"""
Generating some figures using a pretrained SAVi model and the corresponding predictor
"""

import os

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch

from base.baseFigGenerator import BaseFigGenerator

from data.load_data import unwrap_batch_data
from lib.arguments import get_generate_figs_pred
from lib.logger import print_
from lib.metrics import MetricTracker
import lib.utils as utils
from lib.visualizations import add_border, make_gif, visualize_qualitative_eval, visualize_aligned_slots, \
    visualize_tight_row, masks_to_rgb, idx_to_one_hot, overlay_segmentations, COLORS

disp = None
NUM_CONTEXT = None


def get_data_dependent_params(db):
    global disp
    global NUM_CONTEXT
    global NUM_PREDS
    global NUM_SEQS
    if db == "MoviA":
        disp = [0, 1, 2, 3, 4, 7, 11, 14, 17]
        NUM_CONTEXT = 6
    else:
        disp = [0, 1, 2, 3, 4, 9, 14, 19, 24]
        NUM_CONTEXT = 5
    return


class FigGenerator(BaseFigGenerator):
    """
    Class for generating some figures using a pretrained object-centric video prediction model
    """

    def __init__(self, exp_path, savi_model, checkpoint, name_predictor_experiment,
                 num_seqs=30, num_preds=25):
        """
        Initializing the trainer object
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_predictor_experiment)
        super().__init__(
                exp_path=self.exp_path,
                savi_model=savi_model,
                num_seqs=num_seqs
            )
        self.checkpoint = checkpoint
        self.name_predictor_experiment = name_predictor_experiment
        get_data_dependent_params(db=self.exp_params["dataset"]["dataset_name"])
        self.num_seqs = num_seqs
        self.num_preds = num_preds

        self.pred_name = self.exp_params["model"]["predictor"]["predictor_name"]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_pred_{self.pred_name}_{name_predictor_experiment}",
                f"{checkpoint[:-4]}_NumPreds={num_preds}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        if NUM_CONTEXT is not None and self.num_preds is not None:
            self.exp_params["training_prediction"]["num_context"] = NUM_CONTEXT
            self.exp_params["training_prediction"]["num_preds"] = self.num_preds
            self.exp_params["training_prediction"]["sample_length"] = NUM_CONTEXT + self.num_preds
        super().load_data()
        return

    @torch.no_grad()
    def generate_figs(self):
        """
        Evaluating model epoch loop
        """
        utils.set_random_seed()
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        video_length = self.exp_params["training_prediction"]["sample_length"]

        metric_tracker = MetricTracker(exp_path=None, metrics=["psnr", "lpips"])

        for idx in tqdm(range(self.num_seqs)):
            batch_data = self.test_set[idx]
            videos, targets, condition, initializer_data = unwrap_batch_data(self.exp_params, batch_data)
            videos, targets = videos.unsqueeze(0).to(self.device), targets.unsqueeze(0).to(self.device)
            if condition is not None:
              condition = condition.unsqueeze(0).to(self.device)
            initializer_data = {k: v.unsqueeze(0) for k, v in initializer_data.items() if torch.is_tensor(v)}

            n_frames = videos.shape[1]
            if n_frames < num_context + num_preds:
                raise ValueError(f"Seq. length {n_frames} smaller that {num_context = } + {num_preds = }")

            # forward pass through object-centric prediction model
            out_model = self.forward_pass(videos, condition, video_length, initializer_data)
            pred_imgs, pred_recons, pred_masks, individual_recons_history, masks_history = out_model

            # computing metrics for sequence to visualize
            metric_tracker.reset_results()
            metric_tracker.accumulate(
                    preds=pred_imgs.clamp(0, 1),
                    targets=targets[:1, num_context:num_context+num_preds].clamp(0, 1)
                )
            metric_tracker.aggregate()
            results = metric_tracker.get_results()
            psnr, lpips = results["psnr"]["mean"], results["lpips"]["mean"]
            cur_dir = f"img_{idx+1}_psnr={round(psnr,2)}_lpips={round(lpips, 3)}"

            # generating and saving visualizations
            self.compute_visualization(
                    videos=videos,
                    targets=targets,
                    obj_recons_history=individual_recons_history,
                    masks_history=masks_history,
                    pred_imgs=pred_imgs,
                    pred_objs=pred_recons,
                    pred_masks=pred_masks,
                    img_idx=idx,
                    cur_dir=cur_dir
                )
        return

    @torch.no_grad()
    def forward_pass(self, videos, condition, video_length, initializer_data):
        """
        Forward pass through SAVi and Preditor
        """
        B, L, C, H, W = videos.shape
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        out_model = self.model(videos, num_imgs=video_length, **initializer_data)
        slot_history, recons_history, individual_recons_history, masks_history = out_model

        # predicting future slots
        pred_slots = self.predictor(slot_history) if condition is None else self.predictor(slot_history, condition)
        # decoding predicted slots into predicted frames
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)

        return pred_imgs, pred_recons, pred_masks, individual_recons_history, masks_history

    def compute_visualization(self, videos, targets, obj_recons_history, masks_history, pred_imgs,
                              pred_objs, pred_masks, img_idx, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        videos: torch Tensor
            Videos sequence from the dataset, containing the seed and target frames.
            Shape is (B, num_frames, C, H, W)
        pred_imgs: torch Tensor
            Predicted video frames. Shape is (B, num_preds, C, H, W)
        pred_objs: torch Tensor
            Predicted objects corresponding to the predicted video frames.
            Shape is (B, num_preds, num_objs, C, H, W)
        pred_masks: torch Tensor
            Predicted object masks corresponding to the objects in the predicted video frames.
            Shape is (B, num_preds, num_objs, 1, H, W)
        img_idx: int
            Index of the visualization to compute and save
        """
        cur_dir = kwargs.get("cur_dir", f"img_{img_idx+1}")
        utils.create_directory(self.plots_path, cur_dir)

        # some hpyer-parameters of the video model
        B = videos.shape[0]
        num_slots = self.model.num_slots
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        seed_imgs = videos[:, :num_context, :, :]
        seed_objs = obj_recons_history[:, :num_context]
        seed_masks = masks_history[:, :num_context]
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]
        targets = targets[:, num_context:num_context+num_preds, :, :]

        # aligned objects (seed and pred)
        seed_objs = add_border(seed_objs * seed_masks, color_name="green", pad=2)[:, :num_context]
        pred_objs = add_border(pred_objs * pred_masks, color_name="red", pad=2)
        pred_objs = pred_objs.reshape(B, num_preds, num_slots, *pred_objs.shape[-3:])
        all_objs = torch.cat([seed_objs, pred_objs], dim=1)
        fig, _ = visualize_aligned_slots(
                all_objs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "aligned_slots.png")
            )
        plt.close(fig)

        # Video predictions
        fig, ax = visualize_qualitative_eval(
                context=seed_imgs[0],
                targets=target_imgs[0],
                preds=pred_imgs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "qual_eval_rgb.png")
            )
        plt.close(fig)
        fig, ax = visualize_qualitative_eval(
                context=seed_imgs[0],
                targets=target_imgs[0],
                preds=pred_imgs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "qual_eval.png")
            )
        plt.close(fig)
        fig, ax = visualize_tight_row(
                frames=videos[0],
                num_context=num_context,
                disp=disp,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_rgb_gt.png")
            )
        plt.close(fig)
        fig, ax = visualize_tight_row(
                frames=pred_imgs[0],
                num_context=num_context,
                disp=disp,
                is_gt=False,
                savepath=os.path.join(self.plots_path, cur_dir, "row_rgb_pred.png")
            )
        plt.close(fig)

        # masks
        seed_masks_categorical = seed_masks[0, :num_context].argmax(dim=1)
        # if len(pred_masks.shape) > 4:
        #     pred_masks = pred_masks[0]
        pred_masks_categorical = pred_masks.argmax(dim=1)
        if len(pred_masks_categorical.shape) == 3:
            pred_masks_categorical = pred_masks_categorical.unsqueeze(1)
        all_masks_categorical = torch.cat([seed_masks_categorical, pred_masks_categorical], dim=0)
        masks_vis = masks_to_rgb(x=all_masks_categorical)[:, 0]
        fig, ax = visualize_tight_row(
                frames=masks_vis,
                num_context=num_context,
                disp=disp,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_masks.png")
            )
        plt.close(fig)

        # overlay masks
        masks_categorical_channels = idx_to_one_hot(x=all_masks_categorical[:, 0])
        disp_overlay = overlay_segmentations(
            videos[0].cpu().detach(),
            masks_categorical_channels.cpu().detach(),
            colors=COLORS,
            alpha=0.6
        )
        fig, ax = visualize_tight_row(
                frames=disp_overlay,
                num_context=num_context,
                disp=disp,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_overlay.png")
            )
        plt.close(fig)

        # Sequence GIFs
        gt_frames = torch.cat([seed_imgs, target_imgs], dim=1)
        pred_frames = torch.cat([seed_imgs, pred_imgs], dim=1)
        make_gif(
                gt_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "gt_GIF_frames.gif"),
                n_seed=1000,
                use_border=True
            )
        make_gif(
                pred_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "pred_GIF_frames.gif"),
                n_seed=num_context,
                use_border=True
            )
        make_gif(
                masks_vis,
                savepath=os.path.join(self.plots_path, cur_dir, "masks_GIF_masks.gif"),
                n_seed=num_context,
                use_border=True
            )
        make_gif(
                disp_overlay,
                savepath=os.path.join(self.plots_path, cur_dir, "overlay_GIF.gif"),
                n_seed=num_context,
                use_border=True
            )

        # Object GIFs
        for obj_id in range(all_objs.shape[2]):
            make_gif(
                    all_objs[0, :, obj_id],
                    savepath=os.path.join(self.plots_path, cur_dir, f"gt_obj_{obj_id+1}.gif"),
                    n_seed=num_context,
                    use_border=True
                )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_generate_figs_pred()
    print_("Generating figures for predictor model...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            savi_model=args.savi_model,
            checkpoint=args.checkpoint,
            name_predictor_experiment=args.name_predictor_experiment,
            num_seqs=args.num_seqs,
            num_preds=args.num_preds,
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_model(exp_path=figGenerator.parent_exp_path)
    print_("Setting up predictor and loading pretrained parameters")
    figGenerator.load_predictor()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
