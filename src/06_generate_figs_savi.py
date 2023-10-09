"""
Generating figures using a pretrained SAVI model
"""

import os

import matplotlib.pyplot as plt
import torch

from base.baseFigGenerator import BaseFigGenerator

from data import unwrap_batch_data
from lib.arguments import get_generate_figs_savi
from lib.logger import print_
import lib.utils as utils
from lib.visualizations import visualize_recons, visualize_decomp


class FigGenerator(BaseFigGenerator):
    """
    Class for generating figures using a pretrained SAVI model
    """

    def __init__(self, exp_path, savi_model, num_seqs=10):
        """
        Initializing the figure generation module
        """
        super().__init__(
                exp_path=exp_path,
                savi_model=savi_model,
                num_seqs=num_seqs
            )

        model_name = savi_model.split('.')[0]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_SaVIModel_{model_name}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        return

    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """
        Computing visualization
        """
        videos, targets, _, initializer_kwargs = unwrap_batch_data(self.exp_params, batch_data)
        videos, targets = videos.to(self.device), targets.to(self.device)
        out_model = self.model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        cur_dir = f"sequence_{img_idx:02d}"
        utils.create_directory(os.path.join(self.plots_path, cur_dir))

        N = min(10, videos.shape[1])
        savepath = os.path.join(self.plots_path, cur_dir, f"Recons_{img_idx+1}.png")
        visualize_recons(
                imgs=videos[0, :N].clamp(0, 1),
                recons=reconstruction_history[0, :N].clamp(0, 1),
                n_cols=10,
                savepath=savepath
            )

        savepath = os.path.join(self.plots_path, cur_dir, f"ReconsTargets_{img_idx+1}.png")
        visualize_recons(
                imgs=targets[0, :N].clamp(0, 1),
                recons=reconstruction_history[0, :N].clamp(0, 1),
                n_cols=10,
                savepath=savepath
            )

        savepath = os.path.join(self.plots_path, cur_dir, f"Objects_{img_idx+1}.png")
        fig, _, _ = visualize_decomp(
                individual_recons_history[0, :N],
                savepath=savepath,
                vmin=0,
                vmax=1,
            )
        plt.close(fig)

        savepath = os.path.join(self.plots_path, cur_dir, f"masks_{img_idx+1}.png")
        fig, _, _ = visualize_decomp(
                masks_history[0][:N],
                savepath=savepath,
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
        plt.close(fig)
        savepath = os.path.join(self.plots_path, cur_dir, f"maskedObj_{img_idx+1}.png")
        recon_combined = masks_history[0][:N] * individual_recons_history[0][:N]
        recon_combined = torch.clamp(recon_combined, min=0, max=1)
        fig, _, _ = visualize_decomp(
                recon_combined,
                savepath=savepath,
                vmin=0,
                vmax=1,
            )
        plt.close(fig)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_generate_figs_savi()
    print_("Generating figures for SAVI...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            savi_model=args.savi_model,
            num_seqs=args.num_seqs
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_model()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
