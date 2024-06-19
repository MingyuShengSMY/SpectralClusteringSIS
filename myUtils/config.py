import os
import torch

from myUtils.Dict2Class import Dict2Class
from myUtils.dataset_config import DatasetConfig


class Config:
    def __init__(self, config: Dict2Class):
        self.config = config

        try:
            self.mode = self.config.mode
        except AttributeError:
            self.mode = "test"

        self.config_file_path = self.config.config_file_path
        self.config_file_name = self.config.config_file_name

        self.config_file_path_name = "/".join(self.config_file_path.split("/")[1:-1])

        self.random_seed_string = f"random_seed-{self.config.random_seed}"
        self.output_root_dir = os.path.join("outputs", self.config_file_path_name, self.config_file_name, self.random_seed_string)
        if self.mode == "test":
            os.makedirs(self.output_root_dir, exist_ok=True)

        self.method_name = self.config.method_name
        self.method_config = self.config.method_config_dict.__dict__

        self.crf_mark = self.config.crf_mark

        self.task = self.config.task

        self.dataset_dir = self.config.dataset_dir
        self.dataset_list = [DatasetConfig(i, self.dataset_dir, self.config, self.output_root_dir) for i in self.config.dataset_list]

        self.log_dir = os.path.join(self.output_root_dir, self.config.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        self.load_model_mark = self.config.load_model_mark

        if len(self.config.model_load_from) == 0:
            load_path = os.path.join(self.config_file_path_name, self.config_file_name, self.random_seed_string, "trained_model")
        else:
            load_path = self.config.model_load_from
        if len(self.config.model_save_dir) == 0:
            save_path = os.path.join(self.config_file_path_name, self.config_file_name, self.random_seed_string)
        else:
            save_path = self.config.model_save_dir

        self.model_load_from = os.path.join("saved_models", load_path)
        self.model_save_dir = os.path.join("saved_models", save_path)

        self.model_input_size = tuple(self.config.model_input_size)
        self.data_augmentation = self.config.data_augmentation
        self.loss_detail_key_list = self.config.loss_detail_key_list
        self.train_frame_left_right = self.config.train_frame_left_right
        self.test_frame_left_right = self.config.test_frame_left_right

        self.epoch_num = self.config.epoch_num
        self.checkpoint_per_epoch_num = self.config.checkpoint_per_epoch_num
        self.early_stop_patience = self.config.early_stop_patience

        self.batch_size = self.config.batch_size
        try:
            self.batch_size_te = self.config.batch_size_te
        except AttributeError:
            self.batch_size_te = self.batch_size
        self.num_workers = self.config.num_workers

        self.optimizer = self.config.optimizer
        self.learning_rate = self.config.learning_rate
        self.lr_beta1 = self.config.lr_beta1
        self.lr_beta2 = self.config.lr_beta2

        self.metrics = self.config.metrics

        self.early_stop_patience = self.config.early_stop_patience

        self.binary_seg_threshold = self.config.binary_seg_threshold

        self.random_seed = self.config.random_seed

        self.gpu_mark = self.config.gpu_mark
        self.device = torch.device("cuda") if self.gpu_mark and torch.cuda.is_available() else torch.device('cpu')

        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(self.random_seed)

        self.torch_generator_cpu = torch.Generator(device=torch.device('cpu'))
        self.torch_generator_cpu.manual_seed(self.random_seed)

        self.verbose = self.config.verbose

        self.save_cache = self.config.save_cache
        self.use_cache = self.config.use_cache


