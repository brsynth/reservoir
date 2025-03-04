# Supporting content for the reservoir Computing paper with Bacteria paper

[![Github Version](https://img.shields.io/github/v/release/brsynth/molecule-signature-paper?display_name=tag&sort=semver&logo=github)](version)
[![Github Licence](https://img.shields.io/github/license/brsynth/molecule-signature-paper?logo=github)](LICENSE.md)

This repository contains code to support the Molecule Signature publication. See citation for details.

## Table of Contents
- [1. Repository structure](#1-repository-structure)
  - [1.1. Datasets](#11-datasets)
  - [1.2. Supporting Notebooks](#12-supporting-notebooks)
  - [1.3. Source code](#13-source-code)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
  - [3.1. Preparing datasets](#31-preparing-datasets)
  - [3.2 Deterministic enumeration](#32-deterministic-enumeration)
  - [3.3. Train generative models](#33-train-generative-models)
  - [3.4. Predict molecules with generative models](#34-predict-molecules-with-generative-models)
- [4. Citation](#4-citation)

## 1. Repository structure

```text
.
├── data       < placeholder for data files >
│   └── ..
├── notebooks  < supporting jupyter notebooks >
│   ├── 1.enumeration_create_alphabets.ipynb
│   ├── 2.enumeration_results.ipynb
│   ├── 3.analysis_alphabets.ipynb
│   ├── 4.generation_evaluation.ipynb
│   ├── 5.generation_recovery.ipynb
│   └── handy.py
└── src        < source code for data preparation and modeling >
    └── paper
        ├── dataset
        └── learning

```
## 2. Installation

The following steps will set up a `signature-paper` conda environment.

0. **Install Conda:**

    The conda package manager is required. If you do not have it installed, you
    can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
    Follow the instructions on the page to install Conda. For example, on
    Windows, you would download the installer and run it. On macOS and Linux,
    you might use a command like:

    ```bash
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```

    Follow the prompts on the installer to complete the installation.

1. **Install dependencies:**

    ```bash
    conda env create -f recipes/environment.yaml
    conda activate signature-paper
    pip install --no-deps -e .
    ```

2. **Download data:**

    Precomputed alphabets, trained generative models and most important datasets are available as a Zenodo archive: <https://doi.org/10.5281/zenodo.5528831>. Extract the files and place them in the `data` directory.
## 3. Usage
## 4. Citation
   
