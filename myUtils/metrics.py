import sys
import time

import torch
import numpy as np
from myUtils.config import Config
from myUtils.Log import CsvLog
from fvcore.nn import flop_count, parameter_count
from myUtils.others import count_parameters

sys.path.append("..")
from dataloader import DatasetLoader, MyDataset


class Metrics:
    def __init__(self, model, config: Config, dataset_list: list[MyDataset], save=True):
        self.model = model
        self.config = config

        self.dataset_list = dataset_list
        self.dataset_dict = {dataset.dataset_name: dataset for dataset in dataset_list}

        self.intersection_sum_class_dict = {}
        self.union_sum_class_dict = {}
        self.class_pixel_num_dict = {}
        self.iou_class_dict = {}
        self.dice_class_dict = {}
        self.total_pixel_num_no_background = {}
        self.total_pixel_num = {}

        self.image_num = {}
        self.time_cost = {}

        self.img_miou = {}
        self.sample_name = {}

        self.class_count = {}

        self.header = ["DatasetName"] + self.config.metrics

        self.all_metrics_dict = {}

        self.dataset_name_list = [dataset.dataset_name for dataset in dataset_list] + ["All"]
        for dataset_name in self.dataset_name_list:
            if dataset_name != "All":
                class_indicator = self.dataset_dict[dataset_name].dataset_config.class_indicator
                self.intersection_sum_class_dict[dataset_name] = {k: 0 for k in class_indicator}
                self.union_sum_class_dict[dataset_name] = {k: 0 for k in class_indicator}
                self.class_pixel_num_dict[dataset_name] = {k: 0 for k in class_indicator}
                self.iou_class_dict[dataset_name] = {k: 0.0 for k in class_indicator}
                self.dice_class_dict[dataset_name] = {k: 0.0 for k in class_indicator}
                self.class_count[dataset_name] = {d_c: 0 for d_c in class_indicator}

            self.total_pixel_num_no_background[dataset_name] = 0
            self.total_pixel_num[dataset_name] = 0

            self.image_num[dataset_name] = 0
            self.time_cost[dataset_name] = 0

            self.img_miou[dataset_name] = []
            self.sample_name[dataset_name] = []

            self.all_metrics_dict[dataset_name] = {k: 0.0 for k in self.header}
            self.all_metrics_dict[dataset_name]["DatasetName"] = dataset_name
            self.all_metrics_dict[dataset_name]["Params"] = count_parameters(self.model)

        self.save = save
        if self.save:
            self.csv_logger = CsvLog(self.config.log_dir, f"metrics", header=self.header)

        self.start_time = 0
        self.end_time = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()

    def get_set_flops(self, return_dict):

        flops, _ = flop_count(self.model, return_dict[0]['x'].to(self.config.device))
        flops = sum(flops.values())
        params = parameter_count(self.model)
        params = sum(params.values())
        for dataset_name in self.dataset_name_list:
            self.all_metrics_dict[dataset_name]["FLOPS"] = flops
            self.all_metrics_dict[dataset_name]["Params"] = params

    def metrics_update(self, pre_seg: np.ndarray, gt_seg: np.ndarray, dataset_name, time_cost, return_dict, batch_idx):
        img_miou = 0
        class_count = 0
        class_indicator = self.dataset_dict[dataset_name].dataset_config.class_indicator
        for class_name_i in class_indicator:
            class_indicator_i = class_indicator[class_name_i][0]

            pre_seg_class_i = pre_seg == class_indicator_i
            gt_seg_class_i = gt_seg == class_indicator_i

            inter_sum = np.sum(np.logical_and(pre_seg_class_i, gt_seg_class_i))
            union_sum = np.sum(pre_seg_class_i) + np.sum(gt_seg_class_i)
            if np.sum(gt_seg_class_i) > 0:
                img_miou += inter_sum / (union_sum - inter_sum)
                class_count += 1
                self.class_count[dataset_name][class_name_i] = 1

            self.intersection_sum_class_dict[dataset_name][class_name_i] += inter_sum
            self.union_sum_class_dict[dataset_name][class_name_i] += union_sum

        img_miou /= class_count

        self.img_miou[dataset_name].append(img_miou)
        self.sample_name[dataset_name].append(return_dict.get('sample_name'))

        self.time_cost[dataset_name] += time_cost
        self.total_pixel_num_no_background[dataset_name] += np.sum(gt_seg != 0)
        self.total_pixel_num[dataset_name] += gt_seg.size
        self.image_num[dataset_name] += 1

    def get_metrics(self, dataset_name):

        self.all_metrics_dict[dataset_name]['Time Cost Per Image'] = self.time_cost[dataset_name] / self.image_num[dataset_name]
        self.all_metrics_dict[dataset_name]['FPS'] = self.image_num[dataset_name] / self.time_cost[dataset_name]

        m_iou = 0
        m_dice = 0
        m_acc = 0

        class_indicator = self.dataset_dict[dataset_name].dataset_config.class_indicator

        for class_name_i in class_indicator:
            intersection_i = self.intersection_sum_class_dict[dataset_name][class_name_i]
            union_i = self.union_sum_class_dict[dataset_name][class_name_i]
            iou_i = (intersection_i / (union_i - intersection_i)) if union_i - intersection_i != 0 else 0
            dice_i = (2 * intersection_i / union_i) if union_i != 0 else 0
            self.iou_class_dict[dataset_name][class_name_i] = iou_i
            self.dice_class_dict[dataset_name][class_name_i] = dice_i
            m_acc += intersection_i
            m_iou += iou_i
            m_dice += dice_i

        m_acc /= self.total_pixel_num[dataset_name]
        m_iou /= sum(self.class_count[dataset_name].values())
        m_dice /= sum(self.class_count[dataset_name].values())

        self.all_metrics_dict[dataset_name]['mIoU'] = m_iou
        self.all_metrics_dict[dataset_name]['Dice'] = m_dice
        self.all_metrics_dict[dataset_name]['Acc'] = m_acc

        if self.save:
            self.csv_logger.log_a(self.all_metrics_dict[dataset_name])

        return self.all_metrics_dict[dataset_name]

    def get_all_metrics(self):

        total_sample_count = sum(self.image_num.values())

        for dataset_name in self.dataset_dict:
            dataset_img_num = self.image_num[dataset_name]
            ratio = dataset_img_num / total_sample_count
            self.all_metrics_dict["All"]['Time Cost Per Image'] += self.all_metrics_dict[dataset_name]['Time Cost Per Image'] * ratio
            self.all_metrics_dict["All"]['mIoU'] += self.all_metrics_dict[dataset_name]['mIoU'] * ratio
            self.all_metrics_dict["All"]['Dice'] += self.all_metrics_dict[dataset_name]['Dice'] * ratio
            self.all_metrics_dict["All"]['Acc'] += self.all_metrics_dict[dataset_name]['Acc'] * ratio

        self.all_metrics_dict["All"]['FPS'] = 1 / self.all_metrics_dict["All"]['Time Cost Per Image']

        if self.save:
            self.csv_logger.log_a(self.all_metrics_dict["All"])
            img_miou_record = {dataset_name_i: torch.from_numpy(np.array(self.img_miou[dataset_name_i])) for
                               dataset_name_i in
                               self.dataset_name_list[:-1]}
            torch.save(img_miou_record, self.config.log_dir + "/" + "single_img_miou.bin")

        return self.all_metrics_dict["All"]






