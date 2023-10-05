"""
Visualization functions
"""

import itertools
from math import ceil
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
import imageio
from torchvision.utils import draw_segmentation_masks
from webcolors import name_to_rgb

from CONFIG import COLORS


def visualize_sequence(sequence, savepath=None,  tag="sequence", add_title=True, add_axis=False, n_cols=10,
                       size=3, font_size=11, n_channels=3, titles=None, tb_writer=None, iter=0, **kwargs):
    """ Visualizing a grid with several images/frames """

    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])
        del kwargs["suptitle"]

    ims = []
    fs = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        fs.append(f)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title, fontsize=font_size)
            else:
                a.set_title(f"Frame {i}", fontsize=font_size)

    # removing axis
    if(not add_axis):
        for row in range(n_rows):
            for col in range(n_cols):
                a = ax[row, col] if n_rows > 1 else ax[col]
                if n_cols * row + col >= n_frames:
                    a.axis("off")
                else:
                    a.set_yticks([])
                    a.set_xticks([])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        if tb_writer is not None:
            img_grid = torch.stack(fs).permute(0, 3, 1, 2)
            tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims


def visualize_attention_sequence(sequence, indices_max_attended, attention_values, savepath=None,
                                 tag="attention_sequence", add_title=True, add_axis=False, n_cols=10,
                                 size=3, font_size=11, attention_type="vanilla", n_channels=3,
                                 titles=None, tb_writer=None, iter=0, **kwargs, ):
    """ Visualizing a grid with several images/frames """
    assert attention_type in ["vanilla", "time", "object"]
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols)

    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    if("suptitle" in kwargs):
        fig.suptitle(kwargs["suptitle"])
        del kwargs["suptitle"]

    ims = []
    fs = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        f = sequence[i].permute(1, 2, 0).cpu().detach().clamp(0, 1)
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        fs.append(f)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title, fontsize=font_size)
            else:
                if attention_type == "vanilla":
                    time, object = indices_max_attended[i]
                    a.set_title(
                            f"Time:{time}\n Obj:{object}\n attn. val:{'%.3f' % round(attention_values[i], 3)}",
                            fontsize=font_size
                        )
                elif attention_type == "time":
                    time = indices_max_attended[i]
                    a.set_title(
                            f"Time:{time}\n attn. val:{'%.3f' % round(attention_values[i], 3)}",
                            fontsize=font_size
                        )
                else:
                    object = indices_max_attended[i]
                    a.set_title(
                            f"Obj:{object}\n attn. val:{'%.3f' % round(attention_values[i], 3)}",
                            fontsize=font_size
                        )

    # removing axis
    if(not add_axis):
        for row in range(n_rows):
            for col in range(n_cols):
                a = ax[row, col] if n_rows > 1 else ax[col]
                if n_cols * row + col >= n_frames:
                    a.axis("off")
                else:
                    a.set_yticks([])
                    a.set_xticks([])

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        if tb_writer is not None:
            img_grid = torch.stack(fs).permute(0, 3, 1, 2)
            tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims


def visualize_recons(imgs, recons, savepath=None,  tag="recons", n_cols=10, tb_writer=None, iter=0):
    """ Visualizing original imgs, recons and error """
    B, C, H, W = imgs.shape
    imgs = imgs.cpu().detach()
    recons = recons.cpu().detach()
    n_cols = min(B, n_cols)

    fig, ax = plt.subplots(nrows=3, ncols=n_cols)
    fig.set_size_inches(w=n_cols * 3, h=3 * 3)
    for i in range(n_cols):
        a = ax[:, i] if n_cols > 1 else ax
        a[0].imshow(imgs[i].permute(1, 2, 0).clamp(0, 1))
        a[1].imshow(recons[i].permute(1, 2, 0).clamp(0, 1))
        err = (imgs[i] - recons[i]).sum(dim=-3)
        a[2].imshow(err, cmap="coolwarm", vmin=-1, vmax=1)
        a[0].axis("off")
        a[1].axis("off")
        a[2].axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        tb_writer.add_images(fig_name=f"{tag}_imgs", img_grid=np.array(imgs), step=iter)
        tb_writer.add_images(fig_name=f"{tag}_recons", img_grid=np.array(recons), step=iter)

    plt.close(fig)
    return


def visualize_img_err(img, reconstructions):
    """ """
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 3, 1)
    plt.title("Target")
    plt.imshow(img[0].cpu().permute(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.title("Prediction")
    plt.imshow(reconstructions[0].cpu().permute(1, 2, 0))
    plt.subplot(1, 3, 3)
    err = (reconstructions[0] - img[0]).pow(2).sum(dim=0).cpu()
    plt.imshow(err, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Error: MSE={round(err.mean().item(), 5)}")

    plt.tight_layout()
    plt.show()
    plt.close()
    return


def visualize_decomp(objs, savepath=None, tag="decomp", vmin=0, vmax=1, add_axis=False,
                     n_cols=10, titles=None, tb_writer=None, iter=0, **kwargs):
    """
    Visualizing object/mask decompositions, having one obj-per-row

    Args:
    -----
    objs: torch Tensor
        decoded decomposed objects or masks. Shape is (B, Num Objs, C, H, W)
    """
    B, N, C, H, W = objs.shape
    n_channels = C
    if B > n_cols:
        objs = objs[:n_cols]
    else:
        n_cols = B
    objs = objs.cpu().detach()

    ims = []
    fs = []
    fig, ax = plt.subplots(nrows=N, ncols=n_cols)
    fig.set_size_inches(w=n_cols * 3, h=N * 3)
    for col in range(n_cols):
        for row in range(N):
            a = ax[col] if N == 1 else ax[row, col]
            f = objs[col, row].permute(1, 2, 0).clamp(vmin, vmax)
            fim = f.clone()
            if(n_channels == 1):
                fim = fim.repeat(1, 1, 3)
            im = a.imshow(fim, **kwargs)
            ims.append(im)
            fs.append(f)

    for col in range(n_cols):
        a = ax[0, col] if N > 1 else ax[col]
        a.set_title(f"#{col+1}")

    # removing axis
    if(not add_axis):
        for col in range(n_cols):
            for row in range(N):
                a = ax[row, col] if N > 1 else ax[col]
                cmap = kwargs.get("cmap", "")
                if cmap == "gray_r":
                    a.set_xticks([])
                    a.set_yticks([])
                else:
                    a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if tb_writer is not None:
        img_grid = torch.stack(fs).permute(0, 3, 1, 2)
        tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=iter)
    return fig, ax, ims


def visualize_ari(tb_writer, pred, target, score, step):
    pred_scaled = pred / torch.max(pred)
    target_scaled = target / torch.max(target)
    # tb_writer.add_image("ground_truth", target_scaled[None,:,:], step)
    tb_writer.add_image("ground_truth", target_scaled[0].reshape(1, 64, 64), step)
    tb_writer.add_image("predition_mask", pred_scaled[0].reshape(1, 64, 64), step)
    tb_writer.add_scalar("ARI score", score, step)
    return


def visualize_evaluation_slots(tb_writer, slots, step, tag="generated objects", n_channels=3):
    # visualize sequence
    fs = []
    for col in range(slots.shape[0]):
        for row in range(slots.shape[1]):
            f = slots[col, row].permute(1, 2, 0).clamp(0, 1)
            if(n_channels == 1):
                f = f[..., 0]
            fs.append(f)

    img_grid = torch.stack(fs).permute(0, 3, 1, 2)
    tb_writer.add_images(fig_name=tag, img_grid=img_grid, step=step)
    return


def visualize_frame_predictions(context_imgs, pred_imgs, target_imgs, tb_writer=None, savepath=None,
                                step=None, tag=""):
    """ Visualizing ground truth video frames, along with the predicted frames """
    num_context = context_imgs.shape[0]
    num_preds = pred_imgs.shape[0]
    N = num_preds + num_context

    fig, ax = plt.subplots(nrows=2, ncols=N)
    fig.set_size_inches(30, 6)
    for i in range(N):
        if i < num_context:
            ax[0, i].imshow(context_imgs[i].cpu().detach().permute(1, 2, 0).clamp(0, 1))
        else:
            ax[0, i].imshow(target_imgs[i-num_context].cpu().detach().permute(1, 2, 0).clamp(0, 1))
            ax[1, i].imshow(pred_imgs[i-num_context].cpu().detach().permute(1, 2, 0).clamp(0, 1))
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        if tb_writer is not None:
            tb_writer.add_images(
                    fig_name=f"{tag}/Target_imgs",
                    img_grid=np.array(target_imgs.cpu().detach()),
                    step=step
                )
            tb_writer.add_images(
                    fig_name=f"{tag}/Pred_imgs",
                    img_grid=np.array(pred_imgs.cpu().detach()),
                    step=step
                )

    plt.close(fig)
    return


def visualize_qualitative_eval(context, targets, preds, savepath=None, context_titles=None,
                               target_titles=None, pred_titles=None, fontsize=16):
    """
    Qualitative evaluation of one example. Simultaneuosly visualizing context, ground truth
    and predicted frames.
    """
    n_context = context.shape[0]
    n_targets = targets.shape[0]
    n_preds = preds.shape[0]

    n_cols = min(10, max(n_targets, n_context))
    n_rows = 1 + ceil(n_preds / n_cols) + ceil(n_targets / n_cols)
    n_rows_pred = 1 + ceil(n_targets / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(w=n_cols*4, h=(n_rows+1)*4)

    context = add_border(x=context, color_name="green", pad=2).permute(0, 2, 3, 1).cpu().detach()
    targets = add_border(x=targets, color_name="green", pad=2).permute(0, 2, 3, 1).cpu().detach()
    preds = add_border(x=preds, color_name="red", pad=2).permute(0, 2, 3, 1).cpu().detach()

    if context_titles is None:
        ax[0, n_cols//2].set_title("Seed Frames", fontsize=fontsize)
    if target_titles is None:
        ax[1, n_cols//2].set_title("Target Frames", fontsize=fontsize)
    if pred_titles is None:
        ax[n_rows_pred, n_cols//2].set_title("Predicted Frames", fontsize=fontsize)

    for i in range(n_context):
        ax[0, i].imshow(context[i].clamp(0, 1))
        if context_titles is not None:
            ax[0, i].set_title(context_titles[i])
    for i in range(n_preds):
        cur_row, cur_col = i // n_cols, i % n_cols
        if i < n_targets:
            ax[1 + cur_row, cur_col].imshow(targets[i].clamp(0, 1))
            if target_titles is not None:
                ax[1 + cur_row, cur_col].set_title(target_titles[i])
        if i < n_preds:
            ax[n_rows_pred + cur_row, cur_col].imshow(preds[i].clamp(0, 1))
            if pred_titles is not None:
                ax[n_rows_pred + cur_row, cur_col].set_title(pred_titles[i])

    for a_row in ax:
        for a_col in a_row:
            a_col.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax


def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    nc, h, w = x.shape[-3:]
    b = x.shape[:-3]

    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((*b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[..., 0, :, :] = color[0]
    px[..., 1, :, :] = color[1]
    px[..., 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[..., c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[..., pad:h+pad, pad:w+pad] = x
    return px


def visualize_aligned_slots(recons_objs, savepath=None, fontsize=16, mult=3):
    """
    Visualizing the reconstructed objects after alignment of slots.

    Args:
    -----
    recons_objs: torch Tensor
        Reconstructed objects (objs * masks) for a sequence after alignment.
        Shape is (num_frames, num_objs, C, H, W)
    """
    T, N, _, _, _ = recons_objs.shape

    fig, ax = plt.subplots(nrows=N, ncols=T)
    fig.set_size_inches((T * mult, N * mult))
    for t_step in range(T):
        for slot_id in range(N):
            ax[slot_id, t_step].imshow(
                    recons_objs[t_step, slot_id].cpu().detach().clamp(0, 1).permute(1, 2, 0),
                    vmin=0,
                    vmax=1
                )
            if t_step == 0:
                ax[slot_id, t_step].set_ylabel(f"Object {slot_id + 1}", fontsize=fontsize)
            if slot_id == N-1:
                ax[slot_id, t_step].set_xlabel(f"Time Step {t_step + 1}", fontsize=fontsize)
            if slot_id == 0:
                ax[slot_id, t_step].set_title(f"Time Step {t_step + 1}", fontsize=fontsize)
            ax[slot_id, t_step].set_xticks([])
            ax[slot_id, t_step].set_yticks([])
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax


def display_alignment_scores(scores, savepath=None):
    """
    Displaying the alignment between slots from consecutive frames

    Args:
    -----
    scores: torch tensor
        pairwise alignemnt scores between slots from consecutive time steps.
        Shape is (num_frames - 1, num_slots, num_slots)
    """
    num_steps, num_slots, _ = scores.shape

    fig, ax = plt.subplots(1, num_steps)
    fig.set_size_inches(num_steps * 3, 3)
    for t in range(num_steps):
        fig, a = plot_score_matrix(
                scores=scores[t],
                fig=fig,
                ax=ax[t],
                title=f"Alignment on Time Steps {t+1}:{t+2}"
            )
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax


def plot_score_matrix(scores, savepath=None, title=None, fig=None, ax=None, cmap=plt.cm.Blues,):
    """
    Nicely plotting the similarity scores between slots from consecutive time steps
    """
    num_slots = scores.shape[0]

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(num_slots * 3, num_slots * 3)
    im = ax.imshow(
            scores.cpu().detach(),
            interpolation='nearest',
            cmap=cmap,
            vmin=0,
            vmax=1
        )
    if title is not None:
        ax.set_title(title)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(num_slots)
    classes = [f"Slots {i+1}" for i in range(num_slots)]
    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.set_yticks(tick_marks, classes)

    # adding labels with scores
    H, W = scores.shape
    for i, j in itertools.product(range(H), range(W)):
        ax.text(
            x=j,
            y=i,
            s=format(scores[i, j], '.3f'),
            horizontalalignment="center",
            color="white" if scores[i, j] == torch.max(scores[i]) else "black"
        )
    ax.set_xlabel('Slots-Set 2')
    ax.set_ylabel('Slots-Set 1')

    if savepath is not None:
        plt.savefig(savepath)
    return fig, ax


def make_gif(frames, savepath, n_seed=4, use_border=False):
    """ Making a GIF with the frames """
    with imageio.get_writer(savepath, mode='I') as writer:
        for i, frame in enumerate(frames):
            frame = torch.nn.functional.interpolate(frame.unsqueeze(0), scale_factor=2)[0]  # HACK
            up_frame = frame.cpu().detach().clamp(0, 1)
            if use_border:
                color_name = "green" if i < n_seed else "red"
                disp_frame = add_border(up_frame, color_name=color_name, pad=2)
            else:
                disp_frame = up_frame
            disp_frame = (disp_frame * 255).to(torch.uint8).permute(1, 2, 0).numpy()
            writer.append_data(disp_frame)


def visualize_metric(vals, start_x=0, title=None, xlabel=None, savepath=None, **kwargs):
    """ Function for visualizing the average metric per frame """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1)
    ax.plot(vals, linewidth=3)
    ax.set_xticks(ticks=np.arange(len(vals)), labels=np.arange(start=start_x, stop=len(vals) + start_x))
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.close(fig)
    return


def one_hot_to_idx(one_hot):
    """
    Converting a one-hot tensor into idx
    """
    idx_tensor = one_hot.argmax(dim=-3).unsqueeze(-3)
    return idx_tensor


def idx_to_one_hot(x):
    """
    Converting from instance indices into instance-wise one-hot encodings
    """
    num_classes = x.unique().max() + 1
    shape = x.shape
    x = x.flatten().to(torch.int64).view(-1,)
    y = torch.nn.functional.one_hot(x, num_classes=num_classes)
    y = y.view(*shape, num_classes)  # (..., Height, Width, Classes)
    y = y.transpose(-3, -1).transpose(-2, -1)  # (..., Classes, Height, Width)
    return y


def overlay_instances(instances, frames=None, colors=COLORS, alpha=1.):
    """
    Overlay instance segmentations on a sequence of images
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors

    # converting instance from one-hot to isntance indices, if necessary
    N, C, H, W = instances.shape
    if C > 1:
        instance = one_hot_to_idx(instances)

    # some preprocessing on images, or adding white canvas
    if frames is None:
        frames = torch.zeros(N, 3, H, W)
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, instance in zip(frames, instances):
        img = overlay_instance(instance, frame, colors, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_instance(instance, img=None, colors=COLORS, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors

    # converting instance from one-hot to isntance indices, if necessary
    C, H, W = instance.shape
    if C > 1:
        instance = one_hot_to_idx(instance)

    # some preprocessing on images, or adding white canvas
    if img is None:
        img = torch.zeros(3, H, W)
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    instance_ids = instance.unique()
    instance_masks = (instance[0] == instance_ids[:, None, None].to(instance.device))
    cur_colors = [colors[idx.item()] for idx in instance_ids]

    img_with_seg = draw_segmentation_masks(
            img,
            masks=instance_masks,
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255


def one_hot_instances_to_rgb(x, num_channels):
    """
    Converting from multi-channel one-hot instance masks to RGB images for visualization
    """
    x = x.float().round()
    masks_merged = x * torch.arange(num_channels, device=x.device).view(1, 1, -1, 1, 1, 1)
    masks_merged = masks_merged.sum(dim=2)
    masks_rgb = instances_to_rgb(masks_merged, num_channels=num_channels).squeeze(2)
    return masks_rgb


def one_hot_to_instances(x):
    """
    Converting from one-hot multi-channel instance representation to single-channel instance mask
    """
    masks_merged = torch.argmax(x, dim=2)
    return masks_merged


def instances_to_rgb(x, num_channels, colors=None):
    """ Converting from instance masks to RGB images for visualization """
    colors = COLORS if colors is None else colors
    img = torch.zeros(*x.shape, 3)
    background_val = x.flatten(-2).mode(dim=-1)[0]
    for cls in range(num_channels):
        color = colors[cls+1] if cls != background_val else "seashell"
        color_rgb = torch.tensor(name_to_rgb(color)).float()
        img[x == cls, :] = color_rgb / 255
    img = img.transpose(-3, -1).transpose(-2, -1)
    return img


def masks_to_rgb(x):
    """ Converting from SAVi masks to RGB images for visualization """

    # we make the assumption that the background is the mask with the most pixels (mode of distr.)
    num_objs = x.unique().max()
    background_val = x.flatten(-2).mode(dim=-1)[0]

    imgs = []
    for i in range(x.shape[0]):
        img = torch.zeros(*x.shape[1:], 3)
        for cls in range(num_objs + 1):
            color = COLORS[cls+1] if cls != background_val[i] else "seashell"
            color_rgb = torch.tensor(name_to_rgb(color)).float()
            img[x[i] == cls, :] = color_rgb / 255
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = imgs.transpose(-3, -1).transpose(-2, -1)
    return imgs


def visualize_tight_row(frames, num_context=5, disp=[0, 1, 2, 3, 4, 9, 14, 19, 24],
                        is_gt=False, savepath=None):
    """
    Visualizing ground truth or predictions in a tight row

    Args:
    -----
    frames: torch tensor
        Frames to visualize
    num_context: int
        Number of seed frames used
    disp: list
        Indices of the prediction frames to visualize. Idx 0 means first prediction frame.
    is_gt: bool
        If True, the given frames correspond to ground truth. Otherwise correspond to predictions
    """
    num_frames_disp = num_context + len(disp)
    frames = frames.clamp(0, 1).cpu().detach()

    fig, ax = plt.subplots(1, num_frames_disp)
    fig.set_size_inches(2 * num_frames_disp, 2)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(num_frames_disp):
        if is_gt and i < num_context:
            ax[i].imshow(frames[i].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            if i == 0:
                ax[i].set_title(f"t={i+1}", fontsize=10)
            else:
                ax[i].set_title(f"{i+1}", fontsize=10)
        elif is_gt and i >= num_context:
            frame_idx = num_context + disp[i - num_context]
            if frame_idx >= len(frames):
                break
            ax[i].imshow(frames[frame_idx].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            ax[i].set_title(f"{frame_idx+1}", fontsize=10)
        elif not is_gt and i < num_context:
            ax[i].imshow(torch.ones_like(frames[i]).permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
            ax[i].spines['bottom'].set_color('white')
            ax[i].spines['top'].set_color('white')
            ax[i].spines['right'].set_color('white')
            ax[i].spines['left'].set_color('white')
        elif not is_gt and i >= num_context:
            frame_idx = disp[i - num_context]
            if frame_idx >= len(frames):
                break
            ax[i].imshow(frames[frame_idx].permute(1, 2, 0).detach().cpu())
            ax[i].set_xticks([], [])
            ax[i].set_yticks([], [])
        else:
            raise ValueError("?")

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.0)
    return fig, ax


def overlay_segmentations(frames, segmentations, colors, num_classes=None, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if num_classes is None:
        num_classes = segmentations.unique().max() + 1
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)

    # trying to always make the background of the 'seashell' color
    background_id = segmentation.sum(dim=(-1, -2)).argmax().item()
    cur_colors = colors[1:].copy()
    cur_colors.insert(background_id, "seashell")

    img_with_seg = draw_segmentation_masks(
            img,
            masks=segmentation.to(torch.bool),
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255


def hypot(a, b):
    """ """
    y = (a ** 2.0 + b ** 2.0) ** 0.5
    return y


def flow_to_rgb(flow, flow_scaling_factor=50):
    """
    Converting from optical flow to RGB
    """
    height, width = flow.shape[-3], flow.shape[-2]
    scaling = flow_scaling_factor / hypot(height, width)
    x, y = flow[..., 0], flow[..., 1]
    motion_angle = np.arctan2(y, x)
    motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
    motion_magnitude = hypot(y, x)
    motion_magnitude = np.clip(motion_magnitude * scaling, 0.0, 1.0)
    value_channel = np.ones_like(motion_angle)
    flow_hsv = np.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
    flow_rgb = colors.hsv_to_rgb(flow_hsv)
    return flow_rgb

#
