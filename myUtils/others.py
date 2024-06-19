import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import random


def get_normed_laplacian_matrix(affinity_matrix: torch.Tensor):
    d_ar = affinity_matrix.sum(dim=-1, keepdim=True)

    diag_mask = torch.eye(affinity_matrix.shape[0], dtype=torch.bool)

    lap_ar = torch.clone(affinity_matrix)

    lap_ar *= -1
    lap_ar[diag_mask] += d_ar.reshape(-1)

    lap_ar_norm = lap_ar / torch.sqrt(d_ar) / torch.sqrt(d_ar.T)
    return lap_ar_norm


def normalize_01_array(array: np.ndarray, axis=None):
    if axis is not None and not isinstance(axis, int):
        axis = tuple(axis)
    array_min = array.min(axis=axis, keepdims=axis is not None)
    array_max = array.max(axis=axis, keepdims=axis is not None)
    delta = array_max - array_min
    if delta.size > 1:
        delta[delta == 0] = 1e-10
    else:
        if delta == 0:
            delta = 1e-10
    result = (array - array_min) / delta
    return result


def normalize_01_tensor(array: torch.Tensor, axis=None):
    if axis is not None and not isinstance(axis, int):
        axis = list(axis)
    array_min = array.amin(dim=axis, keepdim=axis is not None)
    array_max = array.amax(dim=axis, keepdim=axis is not None)
    delta = array_max - array_min
    if delta.nelement() > 1:
        delta[delta == 0] = 1e-10
    else:
        if delta == 0:
            delta = 1e-10
    result = (array - array_min) / delta
    return result


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"1"


def load_config_file(config_file, save_path_name=True):
    if save_path_name:
        print(f"Loading Config File: '{config_file}'")
    config = json.load(open(config_file))
    if save_path_name:
        config['config_file_path'] = config_file
        config['config_file_name'] = ".".join(os.path.split(config_file)[1].split(".")[:-1])
    return config


def count_parameters(model: torch.nn.Module, train_param=False):
    return sum(p.numel() for p in model.parameters() if not train_param or p.requires_grad)


def get_color_array_recur(all_h_list, color_list, idx1, idx2):
    idx3 = (idx1 + idx2) // 2
    if idx3 == idx1:
        return
    else:
        color_list.append(all_h_list[idx3])
        get_color_array_recur(all_h_list, color_list, idx1, idx3)
        get_color_array_recur(all_h_list, color_list, idx3, idx2)
        return


def resize_label_map(label_map: torch.Tensor, h_w):
    label_map = label_map.squeeze().long()
    one_hot_map = F.one_hot(label_map).float().permute(2, 0, 1).unsqueeze(0)
    one_hot_map_resized = F.interpolate(one_hot_map, size=h_w, mode="bilinear")
    new_label_map = torch.argmax(one_hot_map_resized, dim=1, keepdim=True).long().squeeze()
    return new_label_map






