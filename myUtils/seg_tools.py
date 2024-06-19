import cv2
import numpy as np
import torch
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage import morphology
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

from dataloader import MyDataset
from myUtils.config import Config
# import denseCRF
# from myUtils.others import *
from copy import deepcopy
from scipy.sparse.linalg import eigsh
# from postprocessing.crf import dense_crf
import denseCRF
from PIL import Image, ImageFilter


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


LABEL2RGB_COLOR_MAP = color_map(normalized=True).tolist()


class SegTool:
    def __init__(self, config: Config, dataset: MyDataset):
        self.config = config
        self.dataset = dataset
        self.dataset_config = self.dataset.dataset_config
        self.binary_seg_threshold = self.config.binary_seg_threshold

        self.cluster_k = self.config.method_config.get("cluster_k")
        self.class_n = self.dataset_config.class_n

        self.pre_real_array = None
        self.pre_real_count = None
        self.pre_real_match = None

        if self.cluster_k is not None and self.cluster_k > 0:
            self.auto_cluster_k = False
            self.pre_real_array = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_count = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_match = torch.zeros(size=[self.cluster_k, 2], dtype=torch.int64, device=self.config.device)
        else:
            self.auto_cluster_k = True

        self.crf_mark = self.config.crf_mark

        self.crf_param = (10, 80, 13, 3, 3, 5.0)

    def _update_pre_real_array(self, pre_seg: torch.Tensor, gt_seg: torch.Tensor):
        pre = pre_seg.flatten().long().to(self.config.device)
        gt = gt_seg.flatten().long().to(self.config.device)

        if not self.auto_cluster_k:
            # pass
            self.pre_real_array *= 0
            self.pre_real_count *= 0
        else:
            self.cluster_k = pre.max() + 1
            self.pre_real_array = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_count = torch.zeros(size=[self.cluster_k, self.class_n], device=self.config.device)
            self.pre_real_match = torch.zeros(size=[self.cluster_k, 2], dtype=torch.int64, device=self.config.device)

        for cluster_i in range(self.cluster_k):
            for class_i in range(self.class_n):
                pre_mask = pre == cluster_i
                gt_mask = gt == class_i
                inter_mask = torch.logical_and(pre_mask, gt_mask)
                uni_mask = torch.logical_or(pre_mask, gt_mask)
                self.pre_real_array[cluster_i, class_i] += inter_mask.sum()
                self.pre_real_count[cluster_i, class_i] += uni_mask.sum()
        a = 0

    def get_match(self):
        """
        Hungarian Match
        :return:
        """
        pre_real_iou = self.pre_real_array / self.pre_real_count
        assignments = torch.argmax(pre_real_iou, dim=1)
        for i in range(self.cluster_k):
            self.pre_real_match[i, 0] = i
            self.pre_real_match[i, 1] = assignments[i]
        a = 0

    def deep_spectral_binary_postprocessing(self, return_dict, idx):
        pre_seg = return_dict["pre_seg"][idx]
        gt_seg = return_dict["gt"][idx].to(self.config.device)
        image = return_dict["origin_x"][0].to(self.config.device)

        self._update_pre_real_array(pre_seg, gt_seg)
        self.get_match()

        good_pre_seg = torch.zeros_like(pre_seg)

        for i, j in self.pre_real_match:
            good_pre_seg[pre_seg == i] = j

        return good_pre_seg

    def deep_spectral_multi_class_postprocessing(self, return_dict, idx):
        pre_seg = return_dict["pre_seg"][idx]
        gt_seg = return_dict["gt"][idx].to(self.config.device)
        image = return_dict["origin_x"][0].to(self.config.device)

        self._update_pre_real_array(pre_seg, gt_seg)
        self.get_match()

        good_pre_seg = torch.zeros_like(pre_seg)

        for i, j in self.pre_real_match:
            good_pre_seg[pre_seg == i] = j

        return good_pre_seg

    def seg_postprocessing(self, return_dict_i):
        for i in range(len(return_dict_i["pre_seg"])):
            return_dict_i["pre_seg"][i] = self.deep_spectral_binary_postprocessing(return_dict_i, i)

