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
from myUtils.others import *
from copy import deepcopy
from scipy.sparse.linalg import eigsh
# from postprocessing.crf import dense_crf
import denseCRF
from PIL import Image, ImageFilter


CRF_PARAM = (10, 80, 13, 3, 3, 5.0)


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


def dense_crf(image, pre_seg):
    image = (image * 255).to(torch.uint8)
    # one_hot_pre_seg = F.one_hot(good_pre_seg.squeeze(), num_classes=self.class_n).float().permute(2, 0, 1)
    one_hot_pre_seg = F.one_hot(pre_seg.squeeze()).float()

    # crf_pre_seg = torch.from_numpy(dense_crf(image, one_hot_pre_seg).argmax(0)).long().to(self.config.device).unsqueeze(-1)
    crf_pre_seg = torch.from_numpy(
        denseCRF.densecrf(image.cpu(), one_hot_pre_seg.cpu(), CRF_PARAM)).long()
    return crf_pre_seg


def get_sift_obvious_kp(img: np.ndarray, sigma=0.08, origin_img: np.ndarray = None):
    h, w = img.shape
    max_distance = np.sqrt(h * h + w * w)
    min_distance = 0

    sift = SIFT()

    try:
        sift.detect_and_extract(img)
        kp = sift.keypoints
    except RuntimeError:
        return np.inf

    if len(kp) == 1:
        return np.inf

    agg_cluster = AgglomerativeClustering(n_clusters=None, linkage="single", metric="euclidean", distance_threshold=(min_distance + max_distance)*sigma)
    # agg_cluster = AgglomerativeClustering(n_clusters=None, linkage="single", metric="cosine", distance_threshold=max_distance * 0.05)

    label_kp = agg_cluster.fit_predict(kp)
    # label_kp = agg_cluster.fit_predict(np.concatenate([eigen_vector_copy[kp_i[0], kp_i[1], :].reshape(1, -1) for kp_i in kp], axis=0))

    cluster_n = label_kp.max() + 1

    cluster_coord_list = []

    for i in range(cluster_n):
        cluster_i_coord = kp[label_kp == i, :].mean(0).round().astype(np.int64)
        cluster_coord_list.append(cluster_i_coord.reshape(1, -1))

    cluster_coord_list = np.concatenate(cluster_coord_list, axis=0)

    if origin_img is None:
        pass
    else:
        origin_img_draw = origin_img.copy()
        size_delta_h, size_delta_w = origin_img.shape[0] / img.shape[0], origin_img.shape[1] / img.shape[1]
        cluster_coord_list *= np.array([[size_delta_h, size_delta_w]]).round().astype(np.int64)
        for kp in cluster_coord_list:
            cv2.drawMarker(origin_img_draw, tuple(kp.tolist()), color=(0, 255, 0), markerType=3, markerSize=20)
        cv2.imshow("img", origin_img_draw)
        cv2.waitKey()

    return cluster_n


# def get_sift_obvious_kp(img: np.ndarray, eigen_vector: np.ndarray, sigma=0.08, origin_img: np.ndarray = None):
#     h, w = img.shape
#     max_distance = np.sqrt(h * h + w * w)
#     min_distance = 0
#     # eigen_vector_copy = eigen_vector.copy()
#
#     # eigen_vector_copy = eigen_vector_copy / np.linalg.norm(eigen_vector_copy, axis=-1, keepdims=True)
#     # max_distance = np.max(eigen_vector_copy.reshape(h*w, -1) @ eigen_vector_copy.reshape(-1, h*w))
#
#     # dist_matrix = np.square(eigen_vector_copy.reshape(h*w, 1, -1) - eigen_vector_copy.reshape(1, h*w, -1)).sum(-1)
#     # dist_matrix[np.eye(h*w, dtype=bool)] = np.nan
#     # max_distance = np.nanmax(dist_matrix)
#     # min_distance = np.nanmin(dist_matrix)
#
#     sift = SIFT()
#
#     sift.detect_and_extract(img)
#     kp = sift.keypoints
#
#     agg_cluster = AgglomerativeClustering(n_clusters=None, linkage="single", metric="euclidean", distance_threshold=(min_distance + max_distance)*sigma)
#     # agg_cluster = AgglomerativeClustering(n_clusters=None, linkage="single", metric="cosine", distance_threshold=max_distance * 0.05)
#
#     label_kp = agg_cluster.fit_predict(kp)
#     # label_kp = agg_cluster.fit_predict(np.concatenate([eigen_vector_copy[kp_i[0], kp_i[1], :].reshape(1, -1) for kp_i in kp], axis=0))
#
#     cluster_n = label_kp.max()
#
#     cluster_init = []
#     cluster_coord_list = []
#
#     for i in range(cluster_n):
#         cluster_i_coord = kp[label_kp == i, :].mean(0).round().astype(np.int64)
#         cluster_coord_list.append(cluster_i_coord.reshape(1, -1))
#         cluster_i = eigen_vector[cluster_i_coord[0], cluster_i_coord[1], :]
#         cluster_init.append(cluster_i.reshape(1, -1))
#
#     cluster_init = np.concatenate(cluster_init, axis=0)
#     cluster_coord_list = np.concatenate(cluster_coord_list, axis=0)
#
#     if origin_img is None:
#         pass
#     else:
#         origin_img_draw = origin_img.copy()
#         size_delta_h, size_delta_w = origin_img.shape[0] / img.shape[0], origin_img.shape[1] / img.shape[1]
#         cluster_coord_list *= np.array([[size_delta_h, size_delta_w]]).round().astype(np.int64)
#         for kp in cluster_coord_list:
#             cv2.drawMarker(origin_img_draw, tuple(kp.tolist()), color=(0, 255, 0), markerType=3, markerSize=20)
#         cv2.imshow("img", origin_img_draw)
#         cv2.waitKey()
#
#     return cluster_n, cluster_init


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

    def binary_erosion_dilation_remove(self, pre_seg: torch.Tensor) -> torch.Tensor:
        pre_seg = pre_seg.cpu().numpy()
        for _ in range(pre_seg.size // 500000):
            pre_seg = morphology.binary_erosion(pre_seg)
        for _ in range(pre_seg.size // 500000):
            pre_seg = morphology.binary_dilation(pre_seg)
        if len(np.unique(pre_seg)) > 1:
            binary_mark = False
            if len(np.unique(pre_seg)) == 2:
                binary_mark = True
                pre_seg = pre_seg.astype(np.bool_)
            pre_seg = morphology.remove_small_objects(pre_seg, min_size=pre_seg.size // 1000)
            if binary_mark:
                pre_seg = pre_seg.astype(np.int64)
        pre_seg = torch.from_numpy(pre_seg).to(self.config.device).long()
        return pre_seg

    def multi_erosion_dilation_remove(self, pre_seg: torch.Tensor) -> torch.Tensor:
        pre_seg = pre_seg.cpu().numpy().astype(np.uint8).squeeze(-1)
        # for _ in range(pre_seg.size // 500000):
        # for _ in range(5):
        #     pre_seg = morphology.erosion(pre_seg)
        #     cv2.erode()
        # for _ in range(pre_seg.size // 500000):
        # for _ in range(5):
        #     pre_seg = morphology.dilation(pre_seg)

        # a = pre_seg.copy() / 1.0

        pre_seg = Image.fromarray(pre_seg.astype(np.uint8))
        filter_size = 9
        mode_filter = ImageFilter.ModeFilter(size=filter_size)
        for _ in range(5):
            pre_seg = pre_seg.filter(mode_filter)

        pre_seg = np.array(pre_seg)
        # if len(np.unique(pre_seg)) > 1:
        #     binary_mark = False
        #     if len(np.unique(pre_seg)) == 2:
        #         binary_mark = True
        #         pre_seg = pre_seg.astype(np.bool_)
        #     pre_seg = morphology.remove_small_objects(pre_seg, min_size=pre_seg.size // 1000)
        #     pre_seg = morphology.remove_small_holes(pre_seg, area_threshold=pre_seg.size // 1000)
        #     if binary_mark:
        #         pre_seg = pre_seg.astype(np.int64)

        # cv2.imshow("1", (a * 255).astype(np.uint8))
        # cv2.imshow("mode", (pre_seg * 255).astype(np.uint8))
        # cv2.waitKey()

        pre_seg = torch.from_numpy(pre_seg).to(self.config.device).long().unsqueeze(-1)
        return pre_seg

    def dense_crf(self, image, good_pre_seg):
        if self.crf_mark:
            image = (image * 255).to(torch.uint8)
            # one_hot_pre_seg = F.one_hot(good_pre_seg.squeeze(), num_classes=self.class_n).float().permute(2, 0, 1)
            one_hot_pre_seg = F.one_hot(good_pre_seg.squeeze(), num_classes=self.class_n).float()

            # crf_pre_seg = torch.from_numpy(dense_crf(image, one_hot_pre_seg).argmax(0)).long().to(self.config.device).unsqueeze(-1)
            crf_pre_seg = torch.from_numpy(denseCRF.densecrf(image.permute(1, 2, 0).cpu(), one_hot_pre_seg.cpu(), self.crf_param)).long().to(self.config.device).unsqueeze(-1)
        else:
            crf_pre_seg = good_pre_seg
        return crf_pre_seg

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

    def update_pre_real_array(self, return_dict):

        for i in return_dict:
            pre_seg = return_dict[i]["pre_seg"]
            gt_seg = return_dict[i]["gt"].to(self.config.device)
            b = len(pre_seg)
            for j in range(b):
                pre_seg_j = pre_seg[j]
                gt_seg_j = gt_seg[j]
                self._update_pre_real_array(pre_seg_j, gt_seg_j)

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
        # self.pre_real_array *= 0
        # self.pre_real_count *= 0

    def get_threshed_binary_seg(self, seg: torch.Tensor):

        if self.binary_seg_threshold is None:
            threshold = 0.5
        else:
            threshold = self.binary_seg_threshold

        threshed_seg = torch.greater(seg, threshold)

        return threshed_seg

    def binary_seg_postprocessing(self, seg: torch.Tensor):
        seg_post_processed = self.get_threshed_binary_seg(seg)
        # if len(torch.unique(seg)) > 1:
        # for i in range(seg_post_processed.shape[0]):
            # seg_post_processed[i] = self.binary_erosion_dilation_remove(seg_post_processed[i])
            # seg_post_processed[i] = self.multi_erosion_dilation_remove(seg_post_processed[i])
        return seg_post_processed

    def power_method_eigen_max(self, affinity_matrix: torch.Tensor):
        eigen_vector = torch.randn(size=affinity_matrix.shape[:-1], device=self.config.device).unsqueeze(-1)
        eigen_vector = F.normalize(eigen_vector, dim=1)
        eigen_vector_old = eigen_vector.detach()
        for _ in range(1000):
            eigen_vector = torch.bmm(affinity_matrix, eigen_vector)
            eigen_vector = F.normalize(eigen_vector, dim=1)

            if torch.norm(eigen_vector - eigen_vector_old, dim=1).mean() < 1e-6:
                break
            eigen_vector_old = eigen_vector.detach()
        eigen_value = (torch.bmm(torch.bmm(affinity_matrix, eigen_vector).transpose(1, 2), eigen_vector) / torch.bmm(eigen_vector.transpose(1, 2), eigen_vector)).reshape(-1)
        return eigen_value, eigen_vector

    def power_method_eigen_second_max(self, affinity_matrix: torch.Tensor):
        eigen_value, eigen_vector = self.power_method_eigen_max(affinity_matrix)
        eigen_vector = eigen_vector.squeeze(-1)
        eigen_vector_max = torch.amax(torch.abs(eigen_vector), dim=1)
        eigen_vector_max_arg = torch.argmax(torch.abs(eigen_vector), dim=1)
        x_row = []
        for i in range(len(affinity_matrix)):
            x_row.append(torch.index_select(affinity_matrix[i], dim=0, index=eigen_vector_max_arg[i]))
        x_row = torch.cat(x_row, dim=0)
        x = 1 / (eigen_value * eigen_vector_max).reshape(-1, 1) * x_row

        affinity_matrix = affinity_matrix - torch.bmm((eigen_value.unsqueeze(-1) * eigen_vector).unsqueeze(-1), x.unsqueeze(-2))

        eigen_value, eigen_vector = self.power_method_eigen_max(affinity_matrix)
        return eigen_value, eigen_vector

    def deep_spectral_binary_postprocessing(self, return_dict, idx):
        pre_seg = return_dict["pre_seg"][idx]
        gt_seg = return_dict["gt"][idx].to(self.config.device)
        image = return_dict["origin_x"][0].to(self.config.device)
        # pre_seg = torch.from_numpy(dense_crf(image, F.one_hot(pre_seg[..., 0].long(), len(torch.unique(pre_seg))).float().permute(2, 0, 1)).argmax(0)).to(self.config.device).int().unsqueeze(-1)

        self._update_pre_real_array(pre_seg, gt_seg)
        self.get_match()

        good_pre_seg = torch.zeros_like(pre_seg)

        for i, j in self.pre_real_match:
            good_pre_seg[pre_seg == i] = j

        # good_pre_seg = self.binary_erosion_dilation_remove(good_pre_seg)
        # crf_pre_seg = self.dense_crf(image, good_pre_seg)

        # good_pre_seg = self.multi_erosion_dilation_remove(good_pre_seg)

        return good_pre_seg

    def deep_spectral_multi_class_postprocessing(self, return_dict, idx):
        pre_seg = return_dict["pre_seg"][idx]
        gt_seg = return_dict["gt"][idx].to(self.config.device)
        image = return_dict["origin_x"][0].to(self.config.device)

        # one_hot_pre_seg = F.one_hot(pre_seg.squeeze(), num_classes=self.class_n).float().permute(2, 0, 1)
        #
        # crf_pre_seg = torch.from_numpy(dense_crf(image, one_hot_pre_seg).argmax(0)).long().to(self.config.device)

        self._update_pre_real_array(pre_seg, gt_seg)
        self.get_match()

        good_pre_seg = torch.zeros_like(pre_seg)

        for i, j in self.pre_real_match:
            good_pre_seg[pre_seg == i] = j

        # good_pre_seg = self.multi_erosion_dilation_remove(good_pre_seg)
        # crf_pre_seg = self.dense_crf(image, good_pre_seg)

        return good_pre_seg

    def seg_postprocessing(self, return_dict_i):
        if self.config.method_name in ["AGSD-image", "AGSD-video"]:
            if self.config.task == "binary":
                # for frame_idx in return_dict:
                #     return_dict_i = return_dict[frame_idx]
                return_dict_i["pre_seg"] = self.binary_seg_postprocessing(return_dict_i["pre_seg_prob"])
            else:
                raise ValueError(f"Unknown task {self.config.task}")
        elif self.config.method_name in LABEL_FREE_METHOD_LIST:
            # if self.config.task == "binary":
                # for frame_idx in return_dict:
                # return_dict_i = return_dict[frame_idx]
            for i in range(len(return_dict_i["pre_seg"])):
                return_dict_i["pre_seg"][i] = self.deep_spectral_binary_postprocessing(return_dict_i, i)
            # elif self.config.task == "parts":
            #     # for frame_idx in return_dict:
            #     #     return_dict_i = return_dict[frame_idx]
            #     for i in range(len(return_dict_i["pre_seg"])):
            #         return_dict_i["pre_seg"][i] = self.deep_spectral_multi_class_postprocessing(return_dict_i, i)
            # elif self.config.task == "types":
            #     # for frame_idx in return_dict:
            #     #     return_dict_i = return_dict[frame_idx]
            #     for i in range(len(return_dict_i["pre_seg"])):
            #         return_dict_i["pre_seg"][i] = self.deep_spectral_multi_class_postprocessing(return_dict_i, i)
            # elif self.config.task == "semantic":
            #     # for frame_idx in return_dict:
            #     #     return_dict_i = return_dict[frame_idx]
            #     for i in range(len(return_dict_i["pre_seg"])):
            #         return_dict_i["pre_seg"][i] = self.deep_spectral_multi_class_postprocessing(return_dict_i, i)
            # else:
            #     raise ValueError(f"Unknown task {self.config.task}")
        else:
            raise ValueError(f"Unknown method {self.config.method_name}")
