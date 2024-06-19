# SpecClusteringSIS
Code for my paper 2024.07

[Paper](https_link) and [Video Demo](https_link).

![ ](https://github.com/Finspire13/AGSD-Surgical-Instrument-Segmentation/blob/master/plot.png)

## Setup
* Recommended Environment: Python 3.10, Cuda 12.0+, PyTorch 2.2.0+
* Install dependencies: `pip3 install -r requirements.txt`.

## Data
 1. Download our data for EndoVis 2017 from [Baidu Yun](https://pan.baidu.com/s/1qDq38oiO7DunwVYYNQ_dSQ) (PIN:m0o7) or [Google Drive](https://drive.google.com/file/d/1URJGJGEp1VgtKVMM3gPkie6x69uuE3eZ/view?usp=share_link).
 2. Unzip the file and put into the current directory.
 3. The data includes following sub-directories:

`image`  : Raw images (Left frames) from the [EndoVis 2017 dataset](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) 

`ground_truth`  : Ground truth of binary surgical instrument segmentation.

`cues`  : Hand-designed coarse cues for surgical instruments.

`anchors`  : Anchors generated by fusing cues.

`prediction`  : Final probability maps output by our trained model (Single stage setting).

## Run

Simply run `python3 main.py --config config-endovis17-SS-full.json` .

This config file `config-endovis17-SS-full.json` is for the full model in the single stage setting (SS).

For other experimental settings in our paper, please accordingly modify the config file and the `train_train_datadict`, `train_test_datadict`, `test_datadict` in `main.py` if necessary.

## Output

Results will be saved in a folder named with the `naming` in the config file. 

This output folder will include following sub-directories:

`logs` : A Tensorboard logging file and an numpy logging file.

`models`: Trained models.

`pos_prob`: Probability maps for instruments.

`pos_mask`: Segmentation masks for instruments.

`neg_prob`: Probability maps for non-instruments.

`neg_mask`: Segmentation masks for non-instruments.


## Citation
Liu D. et al. (2020) Unsupervised Surgical Instrument Segmentation via Anchor Generation and Semantic Diffusion. In: Martel A.L. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. Lecture Notes in Computer Science, vol 12263. Springer, Cham. https://doi.org/10.1007/978-3-030-59716-0_63

## License
MIT



