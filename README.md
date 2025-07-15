[![arXiv](https://img.shields.io/badge/arXiv-2305.16404-b31b1b.svg)](https://arxiv.org/abs/2502.06227)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds

<div align="center">
  Lassi Ruoppa<sup>1,*</sup>, Oona Oinonen<sup>1</sup>, Josef Taher<sup>1</sup>, Matti Lehtomäki<sup>1</sup>, 
  Narges Takhtkeshha<sup>2,3</sup>, Antero Kukko<sup>1,4</sup>, Harri Kaartinen<sup>1</sup>, Juha Hyyppä<sup>1,4</sup>
  
  <div style="font-size: smaller; line-height: 1.5; margin-top: 8px;">
    <div><sup>1</sup><i>Department of Remote Sensing and Photogrammetry, Finnish Geospatial Research Institute FGI, The National Land Survey of Finland</i></div>
    <div><sup>2</sup><i>3D Optical Metrology (3DOM) Unit, Bruno Kessler Foundation (FBK)</i></div>
    <div><sup>3</sup><i>Department of Geodesy and Geoinformation, TU Wien</i></div>
    <div><sup>4</sup><i>Department of Built Environment, Aalto University</i></div>
  </div>
</div>
<br><br>

This repository contains the official source code for GrowSP-ForMS, an unsupervised deep learning framework for semantic segmentation of multispectral ALS forest point clouds. The model was introduced in the paper ["Unsupervised deep learning for semantic segmentation of multispectral LiDAR forest point clouds"](https://arxiv.org/abs/2502.06227) and is based on the [GrowSP](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_GrowSP_Unsupervised_Semantic_Segmentation_of_3D_Point_Clouds_CVPR_2023_paper.html) architecture. The multispectral data set associated with the paper is available on [Zenodo](link_here).

<div style="font-size: 10px; line-height: 1.5; margin-top: 8px;">
<sup>*</sup>Corresponding author and maintainer of this repository.
</div>

## Table of contents

<details>
  <summary>Click to expand table of contents</summary>

1. [Installation](#1-installation)

    1.1 [Docker](#11-docker)

    1.2 [Local installation](#12-local-installation)

    1.2.1 [Creating the conda environment](#121-creating-the-conda-environment)

    1.2.2 [Installing CUDA 11.3](#122-installing-cuda-113)

    1.2.3 [Installing gcc 10 and linking to CUDA 11.3](#123-installing-gcc-10-and-linking-to-cuda-113)

    1.2.4 [Installing Minkowski Engine](#124-installing-minkowski-engine)

    1.2.5 [Installing cut pursuit](#125-installing-cut-pursuit)
2. [Data setup](#2-data-setup)

    2.1 [Config and directory structure](#21-config-and-directory-structure)

    2.2 [Using custom data sets](#22-using-custom-data-sets)
3. [Usage](#3-usage)

    3.1 [Data preprocessing](#31-data-preprocessing)

    3.2 [Initial superpoints](#32-initial-superpoints)

    3.3 [Populating missing reflectance values](#33-populating-missing-reflectance-values)

    3.4 [Train the model](#34-train-the-model)

    3.5 [Evaluate a trained model](#35-evaluate-a-trained-model)

    3.6 [Arguments](#36-arguments)
4. [Model weights](#4-model-weights)
5. [Adding new models](#5-adding-new-models)

    5.1 [Simple example](#51-simple-example)

    5.2 [Notes about adding neural network backbones](#52-notes-about-adding-neural-network-backbones)
6. [Raw point cloud format](#6-raw-point-cloud-format)

    6.1 [Supported file types](#61-supported-file-types)

    6.2 [Required format](#62-required-format)
7. [Known issues](#7-known-issues)

    7.1 [GPU Out-Of-Memory during training](#71-gpu-out-of-memory-during-training)
8. [License](#8-license)
9. [Acknowledgements](#9-acknowledgements)

</details>

## 1. Installation

The code has been tested on Ubuntu 20.04 and 22.04. Using Docker is recommended.

### 1.1 Docker

We provide a [`Dockerfile`](Dockerfile) for creating a container with the required libraries installed. To build the docker image, run the following command:
```console
docker build -t growsp_forms_dev .
```
Once built, start the container interactively using e.g. the following:
```console
docker run -it --rm \
    --gpus all \
    --mount type=bind,source=$HOME/<path_to_data>,target=/workspaces/GrowSP-ForMS/data/raw_data/EvoMS \
    --user root \
    growsp_forms_dev
```
where `<path_to_data>` is the local directory containing the EvoMS data set. The above example only mounts the EvoMS data set directory to the container, but this can easily be changed by editing the mounted directories. To enable [developing inside a container](https://code.visualstudio.com/docs/devcontainers/containers) with VSCode, we provide a [`devcontainer.json`](.devcontainer/devcontainer.json) file for convenience. To allow editing of the mounted directories and other arguments without rebuilding the Docker image each time, prebuild the image and modify `devcontainer.json` to use it (see comments in the file).

### 1.2 Local Installation

***This local installation guide has been tested on Ubuntu 22.04. Functionality on other versions can not be guaranteed.***

The below guide details how to install all dependencies locally (not recommended).

#### 1.2.1 Creating the conda environment

Once you have cloned the repository from git, navigate to the directory. Then, the following commands should install all of the dependencies apart from Minkowski Engine
```console
sudo apt install build-essential python3-dev libopenblas-dev
conda env create -f env.yaml
```
The enviroment can then be acticated with
```console
conda activate growsp_forms
```

The [Minkowski Engine](https://nvidia.github.io/MinkowskiEngine/) library is required for sparse convolutions and the installation process can be a bit tricky at times. The library technically only supports CUDA up to 11.1 (though it does work with some newer versions as well). Whichever version you end up using, ensure that it matches the version of your pytorch installation. Below, we install Minkowski Engine with CUDA 11.3.

#### 1.2.2 Installing CUDA 11.3

Run the following command in terminal to check your CUDA version:
```console
nvidia-smi
```
if your version is 11.3 proceed to the next step. If you have any other version, you have to install CUDA 11.3 first. As per the [official Minkowski Engine documentation](https://nvidia.github.io/MinkowskiEngine/overview.html#cuda-11-1-installation), this can be done as follows (**NOTE:** the documentation uses 11.1 while we use 11.3):

```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run --toolkit --silent --override
```

#### 1.2.3 Installing gcc 10 and linking to CUDA 11.3

CUDA 11.3 only works for gcc versions up to 10. Check your gcc version with the following command
```console
gcc --version
```
If your gcc version is higher than 10, you'll have to install gcc 10 and link it with CUDA 11.3. The following steps detail how this can be done

1. Store the maximum gcc version in a variable
```console
export MAX_GCC_VERSION=10
```
2. Install the correct gcc version/verify that you have it installed
```console
sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
```
3. Add symlinks for CUDA 11.3
```console
sudo ln -s /usr/bin/gcc-$MAX_GCC_VERSION /usr/local/cuda/bin/gcc 
sudo ln -s /usr/bin/g++-$MAX_GCC_VERSION /usr/local/cuda/bin/g++
```

#### 1.2.4 Installing Minkowski Engine

Ensure that you have activated the conda environment `evoms`, then run the following command:
```console
export CUDA_HOME=/usr/local/cuda-11.3; pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```
The installation takes a couple of minutes and may display quite a few warnings. Nevertheless, assuming that you've followed the earlier steps, Minkowski Engine should succesfully install.

#### 1.2.5 Installing cut pursuit

The python library for the cut pursuit algorithm has to be compiled from source. For this purpose, we provide a [shell script that should perform the installation semi-automatically](preprocess/cut_pursuit/install.sh). Below we provide a short step by step guide on how to run the script. See [this repository](https://github.com/loicland/cut-pursuit) for the original cut pursuit algorithm code.

1. Navigate to the directory [`/preprocess/cut_pursuit`](preprocess/cut_pursuit/)

2. Ensure that you have installed `awk` and `cmake`

3. Ensure that the value of the variable `CONDAENV` on line 6 of the `install.sh` file corresponds to the path of your conda environment.

4. Ensure that the conda environment is active.

5. Run the script: `./install.sh`

## 2. Data setup

### 2.1 Config and directory structure

Assuming that the name of your data set is `EvoMS`, the GrowSP-ForMS directory should contain the following directories for the program to function correctly:

```
GrowSP-ForMS
|
|—— data
|   |—— raw_data
|   |   |—— EvoMS
|   |   |   |...
|—— cfgs
|   |—— datasets
|   |   |—— EvoMS.yaml
|   |   |—— EvoMS
|   |   |   |—— initial_superpoints.yaml
|   |   |   |—— preprocess.yaml
|   |—— models
|   |   |—— <model_name>.yaml
...
```
place the raw point clouds in the directory `data/raw_data/EvoMS/`. You may use any other name for your data set, simply replace `EvoMS` in the above with the name of your data set.

Regardless of the data set, the data set config file must contain all of the following information (the following is just an example, but the names of the constants must always be the same)
```yaml
NAME: EvoMS # name of the data set
RAW_PATH: data/raw_data/EvoMS # path to raw data
UNPOPULATED_PATH: data/unpopulated_input/EvoMS # path were initial input data with missing reflectance values is stored
INPUT_PATH: data/input/EvoMS # path where final preprocessed input data is stored
SP_PATH: data/initial_superpoints/EvoMS # path where initial superpoints are stored
PSEUDO_PATH: data/pseudo_labels/EvoMS # path where pseudo labels are stored
N_CLASSES: 3 # number of semantic classes
LABELS: { # map from integer semantic labels to class names
  "0": foliage,
  "1": wood
}
IGNORE_LABEL: -1 # label to ignore. Use -1 for None (assuming labels start from 0)
FILENAME_EXTENSION: .las # filename extension of the raw data
```
In addition, the file may contain any data set specific constants used in preprocessing. Note that you must implement the preprocessing sepparately for each data set in the function `preprocess_data` in [`preprocess/preprocess_data.py`](preprocess/preprocess_data.py). The function in question checks the name of the data set specified in the config file and raises a `NotImplementedError` if no method has been implemented for the current data set. The same is true for [`preprocess/create_superpoints.py`](preprocess/create_superpoints.py), where you must implement a method for fetching ground truth labels from the data set.

In case you handle the preprocessing using some other script, the variable `RAW_PATH` is not necessary. Furthermore, if you only plan to evaluate pretrained models, `UNPOPULATED_PATH`, `SP_PATH` and `PSEUDO_PATH` are not strictly necessary either.

* **NOTE:** the parameter `NAME` should match the name of the config file exactly.
* **NOTE:** please follow the described directory and config structure closely. Otherwise the program will most likely crash.

### 2.2 Using custom data sets

By default, only an implementation of the `EvoMS` data set has been added. However, implementing your own data set is relatively straigthforward. Simply create a class named after your data set, e.g. `MyDataset` that inherits from the general data set class `dataset_base`. The only two things you need to implement are the private functions `_init_is_labeled()` and `_get_filenames()`. The former creates a boolean array where index `i` is `True` if the point cloud at index `i` of the data set contains ground truth labels and the latter creates a list of paths that point to the data (`las`, `laz` or `ply` files). The purpose of `_get_filenames()` is to allow using different parts of the data depending on the mode (e.g `train` vs `eval`), while still storing all input data in the same location.

As mentioned in [Section 2.1](#21-config-and-directory-structure), you may add additional parameters to the config file to help with implementing these functionalities. The following code snippet shows how the `MyDataset` class could be implemented:
```python
import numpy as np

from .dataloader import data set_base
from easydict import EasyDict
from .build import data setS


# Remember to register the data set in order to support building from config
@DATASETS.register_module()
class MyDataset(dataset_base):
    def __init__(self, config: EasyDict):
        super().__init__(config)
        # List of filenames corresponding to the current data set
        self._filenames = self._get_filenames()
        # The property is_labeled is a boolean array that should have True at index i, if
        # the file at index i in _filenames has ground truth labels
        self._is_labeled = self._init_is_labeled()


    def _init_is_labeled(self) -> np.ndarray:
        plot_ids = [validate_plot_filename(file)[1] for file in self.filenames]
        is_labeled = np.isin(plot_ids, self.config.LABELED_IDS)
        return is_labeled


    @property
    def is_labeled(self):
        return self._is_labeled
    

    @is_labeled.setter
    def is_labeled(self, new_indices: np.ndarray):
        self._is_labeled = new_indices


    def _get_filenames(self) -> np.ndarray:
        """Get a list of filenames that correspond to the current mode of the dataloader

        Returns:
            np.ndarray: list of filenames
        """
        # Add code here that fetches the filenames
        pass


    @property
    def filenames(self):
        return self._filenames


    @filenames.setter
    def filenames(self, new_names: List[str]):
        self._filenames = new_names
```
* **NOTE:** in the point clouds, ground truth labels should be in a field titled `classification`

## 3. Usage

The program is executed from `main.py`.

### 3.1 Data preprocessing

Set the mode to `data_prepare`.

```console
python main.py --mode=data_prepare -v
```

### 3.2 Initial superpoints

Set the mode to `initial_sp`.

```console
python main.py --mode=initial_sp -v
```

### 3.3 Populating missing reflectance values

Set the mode to `populate_rgb`. Note that the initial superpoints have to be generated before running this mode.

```console
python main.py --mode=populate_rgb -v
```

### 3.4 Train the model

To train the model from scratch (note that setting the mode explicitly is not necessary here, since `train` is the default mode):

```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=train --experiment_name=<some_name>
```

---

To load a model from checkpoint and continue training (see [Section 3.6](#36-arguments) for all supported checkpoint path formats):

```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=train --load_ckpt=<path_to_checkpoint> --experiment_name=<some_name>
```

The configuration files can also be loaded from the checkpoint directory (recommended) by setting the `--cfg_from_ckpt` flag.

### 3.5 Evaluate a trained model

To evaluate a trained model (i.e. compute accuracy metrics for labeled data across the train set):

```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=eval --load_ckpt=<path_to_checkpoint> --experiment_name=<some_name>
```

The corresponding command to evaluate the trained model on the test set is

```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=test --load_ckpt=<path_to_checkpoint> --experiment_name=<some_name>
```

Finally, to evaluate the model for completely unlabeled data, you must also specify the ID of the plot (at least for the EvoMS data set. Feel free to implement this differently for any custom data sets).

```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=eval_unlabeled --load_ckpt=<path_to_checkpoint> --experiment_name=<some_name> --plot_id=<some_id>
```

### 3.6 Arguments

| Argument | Description | Required | Other info | Default |
| --- | --- | --- | --- | --- |
| `--model_name` | Backbone model name | `train`, `eval`, `test` , `eval_unlabeled` | Should match the name of the backbone config yaml file | `ResNet16` |
| `--experiment_name` | Name for the current experiment | `train`, `eval` | Checkpoints etc. are saved in `./ckpt/<dataset_name>/<model_name>/<experiment_name>` | `test` |
| `--dataset_name` | data set name config file name | Always | `NAME` in the data set config file should match `<dataset_name>` | `EvoMS` |
| `--load_ckpt` | Path or name of checkpoint to load model weights from | `eval`, `test` , `eval_unlabeled` | The value can be either an absolute path, a checkpoint file name, an epoch number or `latest`. If the value is just a name, we assume the checkpoint is in the experiment path. If the value is an integer, we search for the checkpoint with the closest epoch from the experiment path and if the value is `latest`, we search for the checkpoint with the highest epoch in said path. | `None` |
| `--cfg_from_ckpt` | Load model and data set configuration files from checkpoint directory. | No | Configuration files are automatically saved in the checkpoint directory when training is started, unless the `--disable_config_dump` flag is set. | `False` |
| `--seed` | Seed for all randomness | No | Some randomness can not be made deterministic even by setting the seed | `42` |
| `-v`, `--verbose` | Print info while running | No | In `train`, `eval`, `test` and `eval_unlabeled` modes all info is always logged (i.e. `-v` has no effect) | `False` |
| `-y`, `--yes` | Automatically answer yes to any prompts | No | Any prompts generally appear at the start of execution and related to overwriting existing checkpoints etc. | `False` |
| `--disable_config_dump` | Disable dumping model config yaml files in experiment path | No | - | `False` |
| `--log_args` | Log args to file | No | - | `False` |
| `--visualize` | Save visualization of predicted classes for each point cloud | No | Only supported in `eval`, `test` and `eval_unlabeled` modes, ignored in any other mode | `False` |
| `--mode` | Which mode to run the program in | Always | Must be one of `train`, `eval`, `test` , `eval_unlabeled`, `data_prepare`, `initial_sp`, `populate_rgb` | `train` |
| `--l_min` | Linearity threshold | No | This argument overrides the linearity threshold defined in the model config. Useful for testing different hyperparameter values for trained models | `None` |
| `--n_overseg_classes` | Number of oversegmentation classes | No | This argument overrides the number of oversegmentation classes defined in the model config. Useful for testing different hyperparameter values for trained models | `None` |
| `--save_latest_only` | Only save the latest model checkpoint during training. | No | By default GrowSP-ForMS creates a new checkpoint file every time semantic primitive clustering is performed. When this flag is set, only the latest checkpoint is saved, and the previous checkpoint is deleted simultaneously. | `False` |

* **NOTE:** in modes other than `train`, `eval`, `test` and `eval_unlabeled`, most arguments are ignored, as they're not relevant to the preprocessing steps.
* **NOTE:** the parser will automatically create paths for loading config files, saving checkpoints and logging. However, these paths are all based on the asumption that the data is stored in the format specified in [Section 2.1](#21-config-and-directory-structure). As such, it's important that the user follows the described directory and config structure, otherwise the program will most likely crash.

## 4. Model weights

The table below provides a download link to pre-trained model weights for the full GrowSP-ForMS model. In addition, we provide links to model and data set configuration files used for the training.

| **Checkpoint** | **Model configuration file** | **Data set configuration file** |
| --- | --- | --- |
| [`full_model_210_ckpt.pth`](https://drive.google.com/file/d/1grRaEznccYtH8cNLCGJMKbUsbUrYCt3Y/view?usp=sharing) | [`ResNet16.yaml`](https://drive.google.com/file/d/1vcdjw-ZLpQ1Miz7Fiau7zF5Y-SbmvkPG/view?usp=sharing) | [`EvoMS.yaml`](https://drive.google.com/file/d/1JdpkTAe67ubn3rDu1pmFLYHdFwby9bwH/view?usp=sharing) |

To evaluate the model using the provided weights, first place the checkpoint and configuration files in the directory `ckpt/EvoMS/ResNet16/full_model` as shown below. If you are using Docker, you may alternatively mount a local directory containing the files to this path:
```
ckpt/EvoMS/ResNet16/full_model
|
|—— cfgs
|   |—— EvoMS.yaml
|   |—— ResNet16.yaml
|—— full_model_210_ckpt.pth
```
Then run:
```console
CUDA_VISIBLE_DEVICES=0, python main.py --mode=test --experiment_name=full_model --load_ckpt=latest --cfg_from_ckpt
```

## 5. Adding new models

In this section we describe how to add new models, including neural network backbones, losses, optimizers and schedulers.

### 5.1 Simple example

Adding a new model is relatively straightforward. Here, we add a new model `ExampleLoss` to illustrate how adding models can be done. `ExampleLoss` is a dummy class that has one required argument `num` and will always return said argument when called, regardless of the input.

1. Create a class `ExampleLoss`. The `__init__()` function should accept only one argument, `config` (type: `EasyDict`). (**NOTE:** if the module is an optimizer, it should also have the argument `base_model`, which is a neural network object and similarly if the model is a scheduler it should also have the argument `optimizer`, which is an optimizer object. See [`models/optimizer.py`](models/optimizer.py) and [`models/scheduler.py`](models/scheduler.py) for examples).

Since `ExampleLoss` is a loss function, we add the code below to the file [`models/loss.py`](models/loss.py). Alternatively, you may also create a completely new file for the loss.
```python
@LOSSES.register_module()
class ExampleLoss(nn.CrossEntropyLoss):
    def __init__(self, config: EasyDict) -> None:
        # Define loss here
        self.num = config.kwargs.num

    def __call__(self, pred, ground_truth):
        return self.num
```
If you want to use an existing loss e.g. from the `torch` library, simply create a wrapper class that inherits from it and call the original class's `__init__()` function inside it. In the below example we show you could add `MSELoss` to your models:
```python
@LOSSES.register_module()
class MSELoss(nn.MSELoss):
    def __init__(self, config: EasyDict) -> None:
        try:
            super().__init__(**config.kwargs)
        except TypeError as e:
            raise TypeError("Error while building loss from config!") from e
```
For more examples, the reader is refered to the code in [`models`](models).

---

2. Define the loss in the model config `yaml` file. The loss should be a dictionary titled `loss` that has two keys, `NAME` and `kwargs`. `NAME` should be the name of the loss class (`ExampleLoss` in this case), and `kwargs` should be a dict of all keyword arguments passed to the loss class. The `yaml` below shows how we would define `ExampleLoss` in the model config file:
```yaml
loss: {
  NAME: ExampleLoss,
  kwargs: {
    num: 0.3
  }
}
```

---

3. Lastly, if you created a new python file for the model you added, import the file in question in [`models/__init__.py`](models/__init__.py). If you added your model to an existing file, steps 1 and 2 are all that is required.

### 5.2 Notes about adding neural network backbones

In the model config, one of the required parameters for neural network backbones is `type`. Currently the only supported value is `sparse`, which refers to any sparse convolution network that uses the Minkowski Engine library. If you add any new type of backbone, use some other value for `type` and define how the input should be passed to the network in the following files [`growsp/evaluate.py`](growsp/evaluate.py), [`growsp/train.py`](growsp/train.py) and [`growsp/growsp_util.py`](growsp/growsp_util.py). At present, any `type` other than `sparse` will raise a `NotImplementedError`. Writing a new collate function may also be necessary.

## 6. Raw point cloud format

### 6.1 Supported file types

Point clouds in the `las/laz` and `ply` formats are supported. There are currently no plans to add support for other file formats.

### 6.2 Required format

The table below lists the required raw point cloud format

| Field | Description | Type | Required |
| --- | --- | --- | --- |
| `x` | x-coordinates | `Number` | Yes |
| `y` | y-coordinates | `Number` | Yes |
| `z` | z-coordinates | `Number` | Yes |
| `red` | red channel (either color or reflectance from scanner 1) | `Number` | Yes |
| `blue` | blue channel (either color or reflectance from scanner 2) | `Number` | Yes |
| `green` | green channel (either color or reflectance from scanner 3) | `Number` | Yes|
| `classification` | ground truth labels. The field must exist even if the point cloud in question does not have actual ground truth labels available. In such a case, the contents may be anything. For `ply` point clouds the input data is expected to contain a field titled `class`, which is then renamed when creating a `PlyData` object (see [`util/pointcloud_io/pointcloud_io.py`](util/pointcloud_io/pointcloud_io.py)). **NOTE:** maximum supported label value is 127 and labels should be non-negative. | `int` | Yes |
| `<field_name>` | Any extra features defined in `dataloader.extra_features` of the model config. By default none are used. | `Number` | If defined in config. No by default |

* **NOTE:** If you do not run the preprocessing code, your data should contain any additional input features you want to use for training the model, but at least `linearity` is required. Furthermore, the point cloud should also contain the fields `x_normals`, `y_normals` and `z_normals`, which represent the estimated surface normals. These values are actually only required if you wish to use PFHs (not default behavior and has to be changed manually), but the dataloader still expects the point clouds to contain the fields (although the values they contain do not affect the training/testing of the model). Alternatively, you can edit the dataloader and collate functions if you wish to remove the requirement for normals. Finally, if your data contains any overlapping regions, each overlapping point should be labeled with an unique id (`overlap_id`) that enables matching them when computing the segmentation accuracy.

## 7. Known issues

### 7.1 GPU Out-Of-Memory during training

When using sparse tensors, the input size between batches can vary significantly. As a result, if some batches are very close to the maximum GPU memory usage, it can happen that the GPU runs out of memory seemingly randomly. We recommend that you choose the batch size such that there's always a bit of extra memory left. This issue is also mentioned in the [Minkowski Engine documentation](https://nvidia.github.io/MinkowskiEngine/issues.html#gpu-out-of-memory-during-training).

## 8. License

This repository contains code under multiple licenses:

- All code for the GrowSP model architecture (in the directory [`growsp`](growsp)) as well as the code for generating superpoints with VCCS (`create_vccs_superpoints` in [`preprocess/create_superpoints.py`](preprocess/create_superpoints.py)) is derived from the code in the [official GrowSP GitHub repository](https://github.com/vLAR-group/GrowSP), which is licensed under the [CC BY-NC-SA 4.0 license](LICENSE-CC-BY-NC-SA-4.0).
- All other code is either original or derived from work under the [MIT license](LICENSE).

Note that due to the licensing, you may **not** use code from [`growsp`](growsp) for commercial purposes, and any derivative work must also be licensed under [CC BY-NC-SA 4.0]((LICENSE-CC-BY-NC-SA-4.0)).

## 9. Acknowledgements

- Majority of the model code is from the original [GrowSP GitHub repository](https://github.com/vLAR-group/GrowSP)
- A lot of code (mainly for model building and logging) has been repurposed from the [PointGPT GitHub repository](https://github.com/CGuangyan-BIT/PointGPT/tree/V1.2)
- The cut pursuit algorihtm used for constructing the intial superpoints is from the official [cut pursuit GitHub repository](https://github.com/loicland/cut-pursuit)
- Code for computing superpoint boundary accuracy metrics the neighborhood graph was repurposed from the [Superpoint Graph GitHub repository](https://github.com/loicland/superpoint_graph)
- Logic for point cloud sparsicification copied from the [SegmentAnyTree GitHub repository](https://github.com/SmartForest-no/SegmentAnyTree)