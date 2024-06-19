import argparse
import time
import numpy as np
from tqdm import tqdm
from myUtils.config import Config, Dict2Class
from myUtils.metrics import Metrics
from myUtils.others import count_parameters, seed_everything
from models import DeepSpectral
from dataloader import DatasetLoader, MyDataset
from myUtils.others import load_config_file
from myUtils.result_saver import ResultSaver
from myUtils.seg_tools import SegTool

# RUN_CHECK = True
RUN_CHECK = False

LOG_SAVE = True
# LOG_SAVE = False

# SAVE_OUTPUT_SEG = True
SAVE_OUTPUT_SEG = False

# ONLY_SAMPLES = True
ONLY_SAMPLES = False


class Method:
    def __init__(self, config: Config):
        self.model = None
        self.optimizer = None
        self.config = config
        self.pre_load_dict = None

        self.device = self.config.device

        self.model = DeepSpectral.DeepSpectral(self.config)

        self.model = self.model.to(self.device)

        print("Parameters: ", count_parameters(self.model))

        self.data_loader = DatasetLoader(self.config)

        self.tr_loader, self.va_loader, self.te_loaders = self.data_loader.get_datasets_dataloaders()

        self.csv_logger_loss = None
        self.npy_logger_loss = None

    def __test_dataset(self, data_loader, dataset: MyDataset, measurer: Metrics):

        batch_range_vis = tqdm(
            data_loader,
            position=1,
            desc=f"Testing on {dataset.dataset_name}",
            leave=False
        )

        result_saver = ResultSaver(self.config, dataset)

        seg_tool = SegTool(self.config, dataset)

        for batch_dict in batch_range_vis:
            s_timer = time.time()
            batch_dict = self.model.get_seg(batch_dict)
            e_timer = time.time()
            b = len(batch_dict[0]["x"])

            for i in batch_dict:
                seg_tool.seg_postprocessing(return_dict_i=batch_dict[i])
                for j in range(b):
                    pre_seg = batch_dict[i]['pre_seg'][j][..., 0].cpu().numpy()
                    gt_seg = batch_dict[i]['gt'][j][0].numpy()

                    if SAVE_OUTPUT_SEG or dataset.samples_file_name == "sample":
                        result_saver.save(batch_dict[i], j)
                    if dataset.samples_file_name != "sample":
                        measurer.metrics_update(pre_seg, gt_seg, dataset.dataset_name, (e_timer - s_timer) / b, batch_dict[i], j)

            if RUN_CHECK:
                break

    def test(self):
        self.model.eval()
        self.model.requires_grad_(False)

        if ONLY_SAMPLES:
            dataloader_list = [i for i in self.te_loaders if i[1].samples_file_name == "sample"]
        else:
            dataloader_list = self.te_loaders

        measurer = Metrics(self.model, self.config, [i[1] for i in self.te_loaders if i[1].samples_file_name != "sample"], save=LOG_SAVE)
        metrics_dict = measurer.all_metrics_dict["All"]

        print_format_count = [max(len(k) + 1, 12) for k in metrics_dict]
        print_format_count[0] = max(print_format_count[0], len(max(self.te_loaders, key=lambda x: len(x[1].dataset_name))[1].dataset_name) + 1)

        print_string = [f"{s:>{print_format_count[i]}}" for i, s in enumerate(metrics_dict.keys())]
        print_string = "|".join(print_string)
        print(print_string)

        for te_loader, te_dataset in dataloader_list:
            self.__test_dataset(te_loader, te_dataset, measurer)

            if te_dataset.samples_file_name != "sample":
                metrics_dict = measurer.get_metrics(te_dataset.dataset_name)
                metric_values_array = np.array(list(metrics_dict.values())[1:])
                metric_values_array = metric_values_array.round(4)
                metrics_values = [metrics_dict["DatasetName"]] + metric_values_array.astype(str).tolist()

                print_string = [f"{s:>{print_format_count[i]}}" for i, s in enumerate(metrics_values)]
                print_string = " ".join(print_string)
                print(print_string)

        if not ONLY_SAMPLES:
            metrics_dict = measurer.get_all_metrics()
            metric_values_array = np.array(list(metrics_dict.values())[1:])
            metric_values_array = metric_values_array.round(4)
            metrics_values = [metrics_dict["DatasetName"]] + metric_values_array.astype(str).tolist()

            print_string = [f"{s:>{print_format_count[i]}}" for i, s in enumerate(metrics_values)]
            print_string = " ".join(print_string)
            print(print_string)


def main():
    args = parser.parse_args()
    config = load_config_file(args.config_file)

    config = Dict2Class(config)
    if args.random_seed <= -1:
        pass
    else:
        config.random_seed = args.random_seed

    config = Config(config)

    seed_everything(config.random_seed)

    method = Method(config)

    method.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config_file', help='path/to/target/config_file.json')
    parser.add_argument('--random_seed', default=-1, help='random seed, this will overwrite the random seed in config files')

    args = parser.parse_args()

    main()
