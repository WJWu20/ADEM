import os
import json
from PIL import Image
import numpy as np
import warnings
import argparse

import torch
from torchvision import transforms
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


warnings.filterwarnings('ignore')

matplotlib.use('agg')

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def denormalize(tensor, means=IM_NORM_MEAN, stds=IM_NORM_STD):
    """Reverses the normalisation on a tensor.
    Performs a reverse operation on a tensor, so the pixel value range is
    between 0 and 1. Useful for when plotting a tensor into an image.
    Normalisation: (image - mean) / std
    Denormalisation: image * std + mean
    Args:
        tensor (torch.Tensor, dtype=torch.float32): Normalized image tensor
    Shape:
        Input: :math:`(N, C, H, W)`
        Output: :math:`(N, C, H, W)` (same shape as input)
    Return:
        torch.Tensor (torch.float32): Demornalised image tensor with pixel
            values between [0, 1]
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    denormalized = tensor.clone()

    for channel, mean, std in zip(denormalized, means, stds):
        channel.mul_(std).add_(mean)

    return denormalized


def scale_and_clip(val, scale_factor, min_val, max_val):
    """
    Helper function to scale a value and clip it within range
    """

    new_val = int(round(val * scale_factor))
    new_val = max(new_val, min_val)
    new_val = min(new_val, max_val)
    return new_val


def visualize_output_and_save(input_, dmp, re_dmp, boxes, save_path, dots=None):
    """
        dots: Nx2 numpy array for the ground truth locations of the dot annotation
            if dots is None, this information is not available
    """
    boxes = boxes.squeeze(0)

    boxes2 = []
    for i in range(0, boxes.shape[0]):
        y1, x1, y2, x2 = int(boxes[i, 1].item()), int(boxes[i, 0].item()), int(boxes[i, 3].item()), int(
            boxes[i, 2].item())
        roi_cnt = dmp[0, y1:y2, x1:x2].sum().item()
        boxes2.append([y1, x1, y2, x2, roi_cnt])

    img1 = input_.permute(1, 2, 0)
    dmp = format_for_plotting(dmp)
    transform = transforms.Compose([
        transforms.Normalize(
            mean=0.5,
            std=0.5
        )
    ])
    re_dmp = transform(re_dmp)
    re_dmp = format_for_plotting(re_dmp)

    # display the input image
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))  # 创建一个包含三个子图的图表
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    ax = axes[0]
    ax.set_axis_off()
    ax.scatter(dots[:, 0], dots[:, 1], c='red', edgecolors='blue')
    ax.set_title("Image", fontsize=20)
    ax.imshow(img1)
    for bbox in boxes2:
        y1, x1, y2, x2, roi_cnt = bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='y', facecolor='none')
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='k', linestyle='--',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(rect2)

    ax = axes[1]
    ax.set_axis_off()
    ax.set_title("Orignal Density Map", fontsize=20)
    ax.imshow(dmp, cmap=plt.cm.jet)

    ax = axes[2]
    ax.set_axis_off()
    ax.set_title("Adaptive Elliptic Density Map", fontsize=20)
    ax.imshow(re_dmp, cmap=plt.cm.jet)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def format_for_plotting(tensor):
    """Formats the shape of tensor for plotting.
    Tensors typically have a shape of :math:`(N, C, H, W)` or :math:`(C, H, W)`
    which is not suitable for plotting as images. This function formats an
    input tensor :math:`(H, W, C)` for RGB and :math:`(H, W)` for mono-channel
    data.
    Args:
        tensor (torch.Tensor, torch.float32): Image tensor
    Shape:
        Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`
        Output: :math:`(H, W, C)` or :math:`(H, W)`, respectively
    Return:
        torch.Tensor (torch.float32): Formatted image tensor (detached)
    Note:
        Symbols used to describe dimensions:
            - N: number of images in a batch
            - C: number of channels
            - H: height of the image
            - W: width of the image
    """

    has_batch_dimension = len(tensor.shape) == 4
    formatted = tensor.clone()

    if has_batch_dimension:
        formatted = tensor.squeeze(0)

    if formatted.shape[0] == 1:
        return formatted.squeeze(0).detach()
    else:
        return formatted.permute(1, 2, 0).detach()


def show_ADEM(im_ids='2091.jpg', save_path=None, data_path='/media/lcc/DATA/wwj/datasets/FSC147/'):
    img = Image.open(os.path.join(data_path, 'images_384_VarV2', im_ids)).convert('RGB')
    density_map1 = np.load(os.path.join(
        'Adaptive_Elliptic_Density_Map',
        im_ids.split('.')[0] + '.npy',
    ))
    density_map1 = torch.from_numpy(density_map1).unsqueeze(0)

    density_map2 = np.load(os.path.join(
        data_path,
        'gt_density_map_adaptive_384_VarV2',
        im_ids.split('.')[0] + '.npy',
    ))
    density_map2 = torch.from_numpy(density_map2).unsqueeze(0)

    img = T.Compose([T.ToTensor()])(img)

    with open(
            os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)
    bboxes1 = torch.tensor(
                annotations[im_ids]['box_examples_coordinates'],
                dtype=torch.float32
            )[:3, [0, 2], :].reshape(-1, 4)[:3, ...]
    points = (
            torch.tensor(annotations[im_ids]['points'])
    ).long()
    if not save_path:
        save_path = os.path.join(im_ids)
    visualize_output_and_save(img, density_map2, density_map1, bboxes1, save_path, dots=points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ADEM parser", add_help=False)
    parser.add_argument('--image_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    show_ADEM(args.image_name, args.save_path, args.data_path)