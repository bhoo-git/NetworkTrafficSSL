## The following repository is for the MCDS Capstone: Improving SSL for Military Applications. This repository is based off of Microsoft's USB: A Unified Semi-supervised learning Benchmark for CV, NLP, and Audio Classification repository. We specifically only include network traffic data related content in this repository. If interested in the original USB repository, refer to the link below for the original repo: `"https://github.com/microsoft/Semi-supervised-learning"`
<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#intro">Introduction</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#benchmark-results">Benchmark Results</a></li>
    <li><a href="#model-zoo">Model Zoo</a></li>
    <li><a href="#contributing">Community</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- Introduction -->

## Introduction

**USB** is a Pytorch-based Python package for Semi-Supervised Learning (SSL). It is easy-to-use/extend, *affordable* to small groups, and comprehensive for developing and evaluating SSL algorithms. USB provides the implementation of 14 SSL algorithms based on Consistency Regularization, and 15 tasks for evaluation from CV, NLP, and Audio domain.

![Code Structure](./figures/code.png)


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

This is an example of how to set up USB locally.
To get a local copy up, running follow these simple example steps.

### Prerequisites

USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

From now on, you can start use USB by typing 

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
```

### Installation

We provide a Python package *semilearn* of USB for users who want to start training/testing the supported SSL algorithms on their data quickly:

```sh
pip install semilearn
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Development

You can also develop your own SSL algorithm and evaluate it by cloning USB:

```sh
git clone https://github.com/microsoft/Semi-supervised-learning.git
```

<p align="right">(<a href="#top">back to top</a>)</p>


### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->

## Usage

USB is easy to use and extend. Going through the bellowing examples will help you familiar with USB for quick use, evaluate an existing SSL algorithm on your own dataset, or developing new SSL algorithms.

### Quick Start with USB package

<!-- TODO: add quick start example and refer lighting notebook -->

Please see [Installation](#installation) to install USB first. We provide colab tutorials for:

- [Beginning example](https://colab.research.google.com/drive/1lFygK31jWyTH88ktao6Ow-5nny5-B7v5)
- [Customize datasets](https://colab.research.google.com/drive/1zbswPm1sM8j0fndUQOeqX2HADdYq-wOw)

### Start with Docker

**Step1: Check your environment**

You need to properly install Docker and nvidia driver first. To use GPU in a docker container
You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)).
Then, Please check your CUDA version via `nvidia-smi`

**Step2: Clone the project**

```shell
git clone https://github.com/microsoft/Semi-supervised-learning.git
```

**Step3: Build the Docker image**

Before building the image, you may modify the [Dockerfile](Dockerfile) according to your CUDA version.
The CUDA version we use is 11.6. You can change the base image tag according to [this site](https://hub.docker.com/r/nvidia/cuda/tags).
You also need to change the `--extra-index-url` according to your CUDA version in order to install the correct version of Pytorch.
You can check the url through [Pytorch website](https://pytorch.org).

Use this command to build the image

```shell
cd Semi-supervised-learning && docker build -t semilearn .
```

Job done. You can use the image you just built for your own project. Don't forget to use the argument `--gpu` when you want
to use GPU in a container.

### Training

Here is an example to train FixMatch on the balanced UNSW-NB15 dataset with 100 labels per class. Training other supported algorithms (on other datasets with different label settings) can be specified by a config file:

```sh
python train.py --c config/usb_network/fixmatch/fixmatch_unsw_img_balanced_1k_0.yaml
```

### Evaluation

After training, you can check the evaluation performance on training logs, or running evaluation script:

```
python eval.py --dataset cifar100 --num_classes 100 --load_path /PATH/TO/CHECKPOINT
```

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Community and Contact

The USB comunity is maintained by:

- Robin Bhoo (rbhoo@andrew.cmu.edu), Carnegie Mellon University

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

We thank the following projects for reference:

- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
- [FixMatch](https://github.com/google-research/fixmatch)
- [CoMatch](https://github.com/salesforce/CoMatch)
- [SimMatch](https://github.com/KyleZheng1997/simmatch)
- [HuggingFace](https://huggingface.co/docs/transformers/index)
- [Pytorch Lighting](https://github.com/Lightning-AI/lightning)
- [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>
