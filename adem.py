import json
import os
import numpy as np
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from astropy.convolution import Gaussian2DKernel
from torch.nn import functional as F
from torchvision.ops import box_convert


def calc_angle(center, b_cod, device='cuda:0'):
    angle = torch.zeros(len(center), device=device)
    ks = -1 * (b_cod[:, 0] - center[:, 0]) / (b_cod[:, 1] - center[:, 1])
    for i, k in enumerate(ks):
        angle[i] = torch.arctan(k)
    angle = torch.nan_to_num(angle, torch.mode(angle).values)
    return angle


def distance(position, min_l, device='cuda:0'):
    alpha = 0.3
    # Calculate distance matrix
    num = len(position)
    distance_matrix = torch.zeros((num, num), device=device)

    for i in range(num):
        for j in range(num):
            distance_matrix[i, j] = torch.sum((position[i, :] - position[j, :]) ** 2)

    sort = torch.argsort(distance_matrix, dim=1)
    idx = sort[:, 1]
    b_cod = position[idx]
    b = torch.sum(torch.sqrt((b_cod - position) ** 2), dim=1)
    b = (torch.clip(b, (1 - alpha) * min_l, (1 + alpha) * min_l) // 2).to(torch.int)
    b = (torch.clip(b, 2, min_l)).to(torch.int)
    sort_points = (b_cod + position) / 2
    return b, b_cod, sort_points


def ellipse2d(center, b, angle, sorted_dots, max_l):
    for i in range(b, max_l):
        if i == max_l - 1:
            a = i
            return a
        for dot in sorted_dots[2:]:
            ellipse = ((dot[0] - center[0]) * torch.cos(angle) + (dot[1] - center[1]) * torch.sin(angle)) ** 2 / (
                    i ** 2) + \
                      ((dot[0] - center[0]) * torch.sin(angle) - (dot[1] - center[1]) * torch.cos(angle)) ** 2 / (
                              b ** 2)
            if ellipse < 1:
                a = i
                return a


def get_a_b_angle(points, scale, min_l, cls_name, device='cuda:0'):
    l = len(points)
    a = torch.zeros(l, device=device)
    b, b_cod, sort = distance(position=points, min_l=min_l, device=device)
    if cls_name == 'candles':
        angle = torch.repeat_interleave(torch.tensor(0), repeats=l)
    else:
        angle = calc_angle(points, b_cod, device=device)
    for i, center in enumerate(points):
        a[i] = ellipse2d(center, b[i], angle[i], sort[i], int(scale * b[i] + 1))
    return a, b, angle


def rotate_gaussian_dmp(img_size, points, a_l, b_l, thetas, device='cuda:0'):
    beta1 = 4.5
    beta2 = 4.5
    n = len(points)
    dmps = torch.zeros((n, img_size[0], img_size[1]), device=device)
    for i, (point, a, b, theta) in enumerate(zip(points, a_l, b_l, thetas)):
        dmp = torch.zeros((img_size[0], img_size[1]), device=device)
        dmp[point[1], point[0]] = 1
        kernel = torch.from_numpy(
            Gaussian2DKernel(x_stddev=a.item() / beta1, y_stddev=b.item() / beta2, theta=theta.item()).array).to(
            torch.float32).unsqueeze(0).to(
            device)
        dmp = F.conv2d(dmp.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0), padding='same')
        if torch.sum(dmp) != 1:
            w = 1 / torch.sum(dmp)
            dmp = w * dmp
        dmps[i] = dmp
    dmps = torch.sum(dmps, dim=0)
    return dmps


def generate_density_map(image_name, data_path, device='cuda:0'):

    save_file = os.path.join(
        f'Adaptive_Elliptic_Density_Map'
    )

    img_class_path = os.path.join(
        data_path,
        f'ImageClasses_FSC147.txt'
    )

    img_classes = {}
    with open(img_class_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            im_id, class_name = line.strip().split('\t')
            img_classes[im_id] = class_name

    if not os.path.isdir(save_file):
        os.makedirs(save_file)

    with open(
            os.path.join(data_path, 'annotation_FSC147_384.json'), 'rb'
    ) as file:
        annotations = json.load(file)

    cls_name = img_classes[image_name]
    ann = annotations[image_name]
    _, h, w = T.ToTensor()(Image.open(os.path.join(
        data_path,
        'images_384_VarV2',
        image_name
    ))).size()

    points = (
            torch.tensor(ann['points'])
    ).long().to(device)
    points[:, 0] = points[:, 0].clip(0, w - 1)
    points[:, 1] = points[:, 1].clip(0, h - 1)

    bboxes = box_convert(torch.tensor(
        ann['box_examples_coordinates'],
        dtype=torch.float32
    )[:3, [0, 2], :].reshape(-1, 4), in_fmt='xyxy', out_fmt='xywh')
    bboxes = bboxes.to(device)
    w_mean = torch.sort(bboxes[:, 2:], dim=1)[0][:, 0].mean()
    scale = (torch.sort(bboxes[:, -2:], dim=-1)[0][:, 1] / torch.sort(bboxes[:, -2:], dim=-1)[0][:, 0]).max()
    a_l, b_l, thetas = get_a_b_angle(points, scale, w_mean, cls_name, device)
    den_map = rotate_gaussian_dmp((h, w), points, a_l, b_l, thetas, device)
    den_map = den_map.detach().cpu().numpy()
    np.save(os.path.join(save_file, os.path.splitext(image_name)[0] + '.npy'), den_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ADEM parser", add_help=False)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_name', type=str, required=True)
    args = parser.parse_args()

    generate_density_map(image_name=args.image_name, data_path=args.data_path)
