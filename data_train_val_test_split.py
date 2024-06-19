import argparse
import glob
import sys

from torch.utils.data import random_split

sys.path.append("")
from myUtils.config import *
from myUtils.others import *
from myUtils.dataset_config import DatasetConfig


class DataSplitter(Config):
    def __init__(self, config: object):
        super().__init__(config)
        self.source_image_name_list = None

    def __split_a_dataset(self, dataset: DatasetConfig):
        """
        :param dataset:
        :param preset_video_name_list: for video dataset, [tr_list, va_list, te_list], tr_list=["video01", "video02"]
                                        if any two or three of the train, val and test have the same video name, the
                                        frames in the video will be split into them according to the split_ratio in
                                        config file
        :return:
        """

        self.source_image_name_list = glob.glob(f"{dataset.originImage_dir}/**/*.*", recursive=True)
        # self.source_gt_image_name_list = glob.glob(f"{dataset.groundTruth_dir}/**/*.*", recursive=True)
        self.source_image_name_list = sorted(self.source_image_name_list)
        # self.source_gt_image_name_list = sorted(self.source_gt_image_name_list)

        self.source_image_name_list = [name_i.split("/")[-2:] for name_i in self.source_image_name_list]
        # self.source_gt_image_name_list = [name_i.split("/")[-2:] for name_i in self.source_gt_image_name_list]
        # self.source_image_name_list = [name_i for name_i in self.source_image_name_list if name_i in self.source_gt_image_name_list]
        source_image_dict = {}
        for video_name, image_name in self.source_image_name_list:
            if source_image_dict.get(video_name):
                source_image_dict[video_name].append(image_name)
            else:
                source_image_dict[video_name] = [image_name]
        self.source_image_name_list = source_image_dict

        video_name_list = list(self.source_image_name_list.keys())

        if len(dataset.set_split_preset) == 0:
            if len(video_name_list) == 1:
                train_video_name_list = video_name_list.copy()
                val_video_name_list = video_name_list.copy()
                test_video_name_list = video_name_list.copy()
            elif len(video_name_list) == 2:
                train_video_name_list = video_name_list[0:1].copy()
                val_video_name_list = video_name_list[0:1].copy()
                test_video_name_list = video_name_list[1:2].copy()
            else:
                train_video_name_list, val_video_name_list, test_video_name_list = (
                    random_split(video_name_list, lengths=dataset.train_val_test_split_ratio, generator=self.torch_generator_cpu))
        else:
            train_video_name_list, val_video_name_list, test_video_name_list = dataset.set_split_preset["train"], dataset.set_split_preset["val"], dataset.set_split_preset["test"]

        # assert len(train_video_name_list), "Train set empty"
        # assert len(val_video_name_list), "Val set empty"
        # assert len(test_video_name_list), "Test set empty"
        if len(set(test_video_name_list + train_video_name_list + val_video_name_list)) != len(video_name_list):
            raise Warning("Some videos may not be used")

        overlapped_video_name = {k: [False, False, False] for k in video_name_list}  # for video dataset
        for video_name in overlapped_video_name:
            if video_name in train_video_name_list:
                overlapped_video_name[video_name][0] = True
            if video_name in val_video_name_list:
                overlapped_video_name[video_name][1] = True
            if video_name in test_video_name_list:
                overlapped_video_name[video_name][2] = True

        train_list = []
        val_list = []
        test_separate_list = []
        test_overlap_list = []  # from same video as train

        for video_name in self.source_image_name_list:
            image_list = self.source_image_name_list[video_name]
            overlapped_list = overlapped_video_name[video_name]
            ratio_list = np.array([ratio if mark else 0 for mark, ratio in zip(overlapped_list, dataset.train_val_test_split_ratio)])
            ratio_list = ratio_list / ratio_list.sum()
            len_list = (len(image_list) * ratio_list).astype(int)
            extra_len = len(image_list) - len_list.sum()
            for mark_i, mark in enumerate(overlapped_list[::-1]):
                mark_i = len(overlapped_list) - mark_i - 1
                if mark:
                    len_list[mark_i] += extra_len
                    break
            len_sum = 0
            for idx, len_i in enumerate(len_list):
                len_sum += len_i
                len_list[idx] = len_sum
            len_list = len_list[:-1]
            train_list_i, val_list_i, test_list_i = np.split(np.array(image_list), len_list)
            for image_name in train_list_i:
                train_list.append(f"{video_name}/{image_name}")
            for image_name in val_list_i:
                val_list.append(f"{video_name}/{image_name}")
            for image_name in test_list_i:
                if overlapped_video_name[video_name][0] and overlapped_video_name[video_name][2]:
                    test_overlap_list.append(f"{video_name}/{image_name}")
                else:
                    test_separate_list.append(f"{video_name}/{image_name}")

        train_list = sorted(train_list)
        val_list = sorted(val_list)
        test_separate_list = sorted(test_separate_list)
        test_overlap_list = sorted(test_overlap_list)
        all_list = sorted(train_list + val_list + test_separate_list + test_overlap_list)
        np.random.seed(1)
        sample_list = np.random.choice(test_separate_list, size=min(50, len(test_separate_list)), replace=False)
        # else:
        #     raise ValueError(f"Unknown data form '{self.data_form}'")

        if len(train_list):
            with open(os.path.join(dataset.input_root_dir, "train_samples.txt"), "w") as f:
                f.write("\n".join(train_list))
        if len(val_list):
            with open(os.path.join(dataset.input_root_dir, "val_samples.txt"), "w") as f:
                f.write("\n".join(val_list))
        if len(test_separate_list):
            with open(os.path.join(dataset.input_root_dir, "test_separate_samples.txt"), "w") as f:
                f.write("\n".join(test_separate_list))
        if len(test_overlap_list):
            with open(os.path.join(dataset.input_root_dir, "test_overlap_samples.txt"), "w") as f:
                f.write("\n".join(test_overlap_list))
        if len(all_list):
            with open(os.path.join(dataset.input_root_dir, "all_samples.txt"), "w") as f:
                f.write("\n".join(all_list))
        if len(sample_list):
            with open(os.path.join(dataset.input_root_dir, "sample_samples.txt"), "w") as f:
                f.write("\n".join(sample_list))

        print(f"Data list text file is generated in {dataset.input_root_dir}")

    def split(self):
        for dataset in self.dataset_list:
            print(f"Splitting {dataset.name} with train_val_test_ratio{dataset.train_val_test_split_ratio}")
            self.__split_a_dataset(dataset)

        print("Data Splitting Done.")


def main(arguments):
    config = load_config_file(arguments.config_file)
    config = Dict2Class(config)

    data_splitter = DataSplitter(config)

    seed_everything(data_splitter.random_seed)

    data_splitter.split()


if __name__ == '__main__':
    os.chdir("")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='path/to/target/config_file.json')

    args = parser.parse_args()

    main(args)
