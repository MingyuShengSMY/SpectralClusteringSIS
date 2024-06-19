import os
import sys

import cv2
import numpy as np
import torch

from dataloader import MyDataset
from myUtils.config import Config
import torchvision.transforms.functional as TF

from myUtils.others import normalize_01_array
from skimage.color import label2rgb

sys.path.append("..")
from myUtils.seg_tools import LABEL2RGB_COLOR_MAP


class ResultSaver:
    def __init__(self, config: Config, dataset: MyDataset):
        self.config = config
        self.dataset = dataset
        self.dataset_config = dataset.dataset_config
        self.class_indicator = self.dataset_config.class_indicator

        self.video_names_list = dataset.video_names_list

        self.output_affinity_matrix_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name,"affinity_matrix")
        os.makedirs(self.output_affinity_matrix_dir, exist_ok=True)

        self.output_eigenvectors_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "eigenvectors")
        os.makedirs(self.output_eigenvectors_dir, exist_ok=True)

        self.output_pre_seg_raw_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name,"pre_seg_raw")
        os.makedirs(self.output_pre_seg_raw_dir, exist_ok=True)

        self.output_pre_seg_color_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "pre_seg_color")
        os.makedirs(self.output_pre_seg_color_dir, exist_ok=True)

        self.output_pre_seg_overlay_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "pre_seg_overlay")
        os.makedirs(self.output_pre_seg_overlay_dir, exist_ok=True)

        self.output_gt_seg_overlay_dir = os.path.join(self.dataset_config.output_dir, self.dataset.samples_file_name, "gt_seg_overlay")
        os.makedirs(self.output_gt_seg_overlay_dir, exist_ok=True)

        self.origin_size = None
        self.origin_size_output = None

        for video_name in self.video_names_list:
            os.makedirs(self.output_pre_seg_raw_dir + "/" + video_name, exist_ok=True)
            os.makedirs(self.output_pre_seg_color_dir + "/" + video_name, exist_ok=True)
            os.makedirs(self.output_affinity_matrix_dir + "/" + video_name, exist_ok=True)
            os.makedirs(self.output_eigenvectors_dir + "/" + video_name, exist_ok=True)
            os.makedirs(self.output_pre_seg_overlay_dir + "/" + video_name, exist_ok=True)
            os.makedirs(self.output_gt_seg_overlay_dir + "/" + video_name, exist_ok=True)

    def save(self, return_dict: dict, idx):

        self.origin_size = return_dict["origin_shape"][idx].cpu().numpy().tolist()
        self.origin_size_output = return_dict["origin_shape_output"][idx].cpu().numpy().tolist()

        if return_dict.get('affinity_matrix') is not None:
            img = return_dict['affinity_matrix'][idx].cpu().numpy()
            img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

            self.save_intermedia_affinity(return_dict['sample_name'][idx], img)

        if return_dict.get('pre_seg_raw') is not None:
            img = return_dict['pre_seg_raw'][idx][..., 0].cpu().numpy()

            self.save_raw_seg(return_dict['sample_name'][idx], img)

        if return_dict.get('eigen_vector') is not None:
            img = return_dict['eigen_vector'][idx].cpu().numpy()

            self.save_intermedia_eigen(return_dict['sample_name'][idx], img)

        image_name = return_dict['sample_name'][idx]

        origin_img = return_dict["origin_x"][idx:idx+1][0]
        gt_seg = return_dict["gt"][idx:idx + 1][0]
        pre_seg = return_dict["pre_seg"][idx:idx + 1].permute(0, 3, 1, 2)[0].long()

        self.save_gt_overlay(image_name, gt_seg.permute(1, 2, 0).cpu().numpy(), origin_img.permute(1, 2, 0).cpu().numpy())
        self.save_pre_overlay(image_name, pre_seg.permute(1, 2, 0).cpu().numpy(), origin_img.permute(1, 2, 0).cpu().numpy())
        self.save_color_result_seg(image_name, return_dict['pre_seg'][idx][..., 0].cpu().numpy())

    def __prepro(self, image):
        image = image.squeeze()

        return image

    def add_border(self, image: np.ndarray):
        if self.dataset_config.prepro_crop and image.shape != self.origin_size:
            if len(image.shape) == 2:
                origin_image = np.zeros(self.origin_size_output)
            else:
                origin_image = np.zeros(self.origin_size_output + [3])

            x1, y1 = self.dataset_config.valid_field_left_top
            h, w = self.dataset_config.valid_field_size
            h = min(image.shape[0], h)
            w = min(image.shape[1], w)
            origin_image[x1: x1 + h, y1: y1 + w] = image
            image = origin_image
        else:
            pass
        return image

    def get_overlay(self, seg: np.ndarray, image: np.ndarray, alpha=0.4):
        seg = self.__prepro(seg)
        image = self.__prepro(image)

        image_seg = self.get_color_seg(seg).astype(np.uint8)[:, :, ::-1]
        image = (image[:, :, ::-1] * 255).astype(np.uint8)
        image_combined = image.copy()
        image_combined[seg != 0, :] = (image_seg[seg != 0, :] * alpha + image[seg != 0, :] * (1 - alpha)).astype(np.uint8)

        image_combined = TF.resize(torch.from_numpy(image_combined).permute(2, 0, 1), size=self.origin_size, interpolation=TF.InterpolationMode.BILINEAR).permute(1, 2, 0).numpy()

        image_combined = self.add_border(image_combined)

        return image_combined

    def save_gt_overlay(self, image_name: str, seg: np.ndarray, image: np.ndarray):
        image_seg = self.get_overlay(seg, image)

        target_path = self.output_gt_seg_overlay_dir + "/" + image_name
        cv2.imwrite(target_path, image_seg)

    def save_pre_overlay(self, image_name: str, seg: np.ndarray, image: np.ndarray):
        image_seg = self.get_overlay(seg, image)

        target_path = self.output_pre_seg_overlay_dir + "/" + image_name
        cv2.imwrite(target_path, image_seg)

    def save_intermedia_affinity(self, array_name: str, array: np.ndarray):

        target_path = self.output_affinity_matrix_dir + "/" + array_name

        cv2.imwrite(target_path, array)

    def save_intermedia_eigen(self, array_name: str, array: np.ndarray):

        target_path_dir = self.output_eigenvectors_dir + "/" + array_name.split(".")[0]
        os.makedirs(target_path_dir, exist_ok=True)

        for i in range(1, array.shape[-1]):
            target_path = target_path_dir + f"/{i}-th_eigenvector.png"
            img = array[:, :, i]
            img = normalize_01_array(img) * 255
            img = img.astype(np.uint8)

            img = cv2.resize(img, dsize=self.origin_size[::-1], interpolation=cv2.INTER_NEAREST)

            img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)

            cv2.imwrite(target_path, img)

    def save_raw_seg(self, image_name: str, image: np.ndarray):

        target_path = self.output_pre_seg_raw_dir + "/" + image_name

        image = self.__prepro(image)

        image_color = (label2rgb(image.astype(np.uint64) + 1, colors=LABEL2RGB_COLOR_MAP[1:])*255)[:, :, ::-1]

        scale = image_color.max()
        scale = 255.0 / scale if scale != 0 else 1

        image_color *= scale

        image_color = image_color.astype(np.uint8)

        cv2.imwrite(target_path, image_color)

    def get_color_seg(self, image):
        image_color = np.zeros(shape=list(image.shape) + [3])

        for class_indicator_name in self.class_indicator:
            class_indicator_i = self.class_indicator[class_indicator_name]
            image_color[image == class_indicator_i[0]] = class_indicator_i[2]
        return image_color

    def save_color_result_seg(self, image_name: str, image: np.ndarray):

        target_path = self.output_pre_seg_color_dir + "/" + image_name

        image = self.__prepro(image)

        # image = resize_label_map(image, self.origin_size)

        image_color = self.get_color_seg(image)

        cv2.imwrite(target_path, image_color[:, :, ::-1])




