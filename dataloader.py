import cv2
import torch.nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
from myUtils.config import *
from myUtils.others import *


class MyDataset(Dataset):
    def __init__(self, config: Config, dataset_config: DatasetConfig, samples_file_name: str, samples_names_list, data_aug, test_frame_mode):
        super().__init__()

        self.config = config

        self.samples_file_name = samples_file_name.replace("_samples", "")

        self.dataset_config = dataset_config
        self.dataset_name = dataset_config.name + "_" + self.samples_file_name

        self.prepro_crop_left_top = dataset_config.valid_field_left_top
        self.prepro_crop_size = dataset_config.valid_field_size
        self.prepro_crop = dataset_config.prepro_crop

        self.frame_left_right = None
        self.test_frame_mode = test_frame_mode

        self.aug_transforms_mark = self.config.data_augmentation and data_aug

        self.source_image_dir_list = []
        self.source_image_path_dict = {}

        self.device = self.config.device

        self.getitem = None

        self.return_dict_keys = None

        self.samples_names_list = samples_names_list
        self.sample_video_image_dict = {}
        self.image_names_list = [i.split("/")[1] for i in self.samples_names_list]
        self.video_names_list = []
        for sample_name in self.samples_names_list:
            video_name, image_name = sample_name.split("/")
            if not self.sample_video_image_dict.get(video_name):
                self.sample_video_image_dict[video_name] = {"len": 1, "image_name_list": [image_name]}
                self.video_names_list.append(video_name)
            else:
                self.sample_video_image_dict[video_name]["len"] += 1
                self.sample_video_image_dict[video_name]["image_name_list"].append(image_name)

        self.set_extra_frames_train_mode(test_frame_mode)

        self.return_dict_keys = ["x", "gt", ]
        self.key_need_prepro = ['x']
        self.image_key = ["x", 'gt']

        self.source_image_dir_list.append(self.dataset_config.originImage_dir)
        self.source_image_dir_list.append(self.dataset_config.groundTruth_dir)

        self.source_image_path_dict['x'] = [f"{self.source_image_dir_list[0]}/{sample_name_i}" for
                                            sample_name_i in samples_names_list]
        self.source_image_path_dict['gt'] = [f"{self.source_image_dir_list[1]}/{sample_name_i}" for
                                             sample_name_i in samples_names_list]

        assert len(self.return_dict_keys) == len(self.source_image_path_dict), "Not all images loaded are used"
        assert len(
            np.unique([len(i) for i in
                       self.source_image_path_dict.values()])) == 1, "The number of source images are different"

    def __len__(self):
        return sum([d["len"] - (self.frame_left_right[1] - self.frame_left_right[0]) for d in self.sample_video_image_dict.values()])

    def set_extra_frames_train_mode(self, train: bool):
        self.frame_left_right = self.config.train_frame_left_right if train else self.config.test_frame_left_right
        for video_name in self.sample_video_image_dict:
            assert self.frame_left_right[1] - self.frame_left_right[0] + 1 <= self.sample_video_image_dict[video_name]["len"]

    def __getitem_image__(self, idx):
        return_dict = {}
        for return_key in self.return_dict_keys:
            if return_key == 'x':
                path_i = self.source_image_path_dict[return_key][idx]
                image = cv2.imread(path_i)
                image = image[:, :, ::-1]
                image = TF.to_tensor(image.copy())

                return_dict[f"{return_key}_path"] = path_i

                return_dict["origin_shape_output"] = torch.from_numpy(np.array(image.shape[1:3]))

                if self.prepro_crop:
                    image = TF.crop(image, *self.prepro_crop_left_top, *self.prepro_crop_size)

                return_dict["origin_shape"] = torch.from_numpy(np.array(image.shape[1:3]))

                if len(self.config.model_input_size):
                    image = TF.resize(image, self.config.model_input_size)

                return_dict["origin_x"] = torch.clone(image.detach())
                image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            elif return_key == 'gt':
                path_i = self.source_image_path_dict[return_key][idx]
                image_ = cv2.imread(path_i, 0)
                image = np.zeros_like(image_)
                for class_key in self.dataset_config.class_indicator:
                    image[image_ == self.dataset_config.class_indicator[class_key][1]] = (
                        self.dataset_config.class_indicator)[class_key][0]
                return_dict[f"{return_key}_path"] = path_i

                image = torch.from_numpy(image.copy())
                image = image.unsqueeze(-1)
                image = image.permute(2, 0, 1)

                if self.prepro_crop:
                    image = TF.crop(image, *self.prepro_crop_left_top, *self.prepro_crop_size)

                if len(self.config.model_input_size):
                    image = TF.resize(image, self.config.model_input_size, interpolation=TF.InterpolationMode.NEAREST_EXACT)

            else:
                raise ValueError(f"Unknown image_key")

            return_dict[return_key] = image

        return_dict["sample_name"] = self.samples_names_list[idx]

        return return_dict

    def __getitem_video__(self, origin_idx):
        idx = origin_idx

        # find the next frame, if use video as input, no use in this study
        ##################################
        idx_cache = 0
        for video_idx, video_name in enumerate(self.video_names_list):
            if idx_cache + self.sample_video_image_dict[video_name]["len"] - (self.frame_left_right[1] - self.frame_left_right[0]) <= origin_idx:
                idx_cache += self.sample_video_image_dict[video_name]["len"] - (self.frame_left_right[1] - self.frame_left_right[0])
                idx += (self.frame_left_right[1] - self.frame_left_right[0])
                idx_cache += (self.frame_left_right[1] - self.frame_left_right[0])
            else:
                break
        idx -= self.frame_left_right[0]
        ##################################3

        return_dict_all_frames = {}

        for i in range(self.frame_left_right[0], self.frame_left_right[1] + 1):
            return_dict = self.__getitem_image__(idx + i)
            return_dict_all_frames[i] = return_dict

        return return_dict_all_frames

    def __getitem__(self, idx):
        return_dict = self.__getitem_video__(idx)
        return return_dict


class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config

        self.tr_dataset_list = []
        self.tr_no_aug_dataset_list = []
        self.va_dataset_list = []
        self.te_dataset_list = []

        for dataset in self.config.dataset_list:
            if self.config.mode == "test" and not dataset.available:
                continue
            for txt_file_path in dataset.use_for_train:
                with open(txt_file_path, "r") as f:
                    samples_names = f.read().split("\n")
                self.tr_dataset_list.append(
                    MyDataset(self.config, dataset, txt_file_path.split("/")[-1].split(".")[0], samples_names, data_aug=True, test_frame_mode=True))
            for txt_file_path in dataset.use_for_val:
                with open(txt_file_path, "r") as f:
                    samples_names = f.read().split("\n")
                self.va_dataset_list.append(
                    MyDataset(self.config, dataset, txt_file_path.split("/")[-1].split(".")[0], samples_names, data_aug=True, test_frame_mode=True))
            for txt_file_path in dataset.use_for_test:
                with open(txt_file_path, "r") as f:
                    samples_names = f.read().split("\n")
                self.te_dataset_list.append(
                    MyDataset(self.config, dataset, txt_file_path.split("/")[-1].split(".")[0], samples_names, data_aug=True, test_frame_mode=False))

        if len(self.tr_dataset_list):
            self.tr_dataset_aug = ConcatDataset(self.tr_dataset_list)
        else:
            self.tr_dataset_aug = self.tr_dataset_list
        if len(self.va_dataset_list):
            self.va_dataset = ConcatDataset(self.va_dataset_list)
        else:
            self.va_dataset = self.va_dataset_list

        self.te_dataset = self.te_dataset_list

        self.batch_size_tr = self.config.batch_size
        self.batch_size_va = self.config.batch_size
        self.batch_size_te = self.config.batch_size_te

        if len(self.tr_dataset_aug):
            self.tr_loader = DataLoader(
                self.tr_dataset_aug,
                batch_size=self.batch_size_tr,
                shuffle=True,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
        else:
            self.tr_loader = self.tr_dataset_aug

        if len(self.va_dataset):
            self.va_loader = DataLoader(
                self.va_dataset,
                batch_size=self.batch_size_va,
                shuffle=False,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            )
        else:
            self.va_loader = self.va_dataset

        self.te_loaders = [
            [DataLoader(
                te_dataset,
                batch_size=self.batch_size_te,
                shuffle=False,
                num_workers=self.config.num_workers,
                generator=torch.Generator().manual_seed(self.config.random_seed)
            ), te_dataset] for te_dataset in self.te_dataset
        ]

    def get_datasets_dataloaders(self):
        return self.tr_loader, self.va_loader, self.te_loaders
