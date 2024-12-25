# Diffusion Transformers for 3D Point Cloud Denoising

This project investigates transformer-based models for 3D point cloud denoising using the P2P-Bridge Diffusion framework. By integrating multi-resolution hash encoding , these models aim to capture multi-scale features more efficiently, offering an alternative to traditional PointNet-based approaches.

## Table of Contents
1. [**Project Overview**](#project-overview)
2. [**Key Contributions**](#key-contributions)
3. [**Requirements**](#requirements)
4. [**Data Preparation**](#data-preparation)  
   * [Object Datasets (PU-Net and PC-Net)](#object-datasets-pu-net-and-pc-net)
5. [**Training**](#training)
6. [**Evaluation**](#evaluation)  
   * [PU-Net and PC-Net](#pu-net-and-pc-net)
7. [**Denoise Objects**](#denoise-objects)  
8. [**Acknowledgements**](#acknowledgements)
   
## Project Overview
We explore transformer-based architectures for point cloud denoising using the P2P-Bridge framework. By incorporating multi-resolution hash encoding (inspired by InstantNGP), we address multi-scale feature representation in a more memory-efficient way than dense grid encodings. This approach seeks to surpass existing PointNet-based methods on multiple datasets.

## Key Contributions
- **Transformer-Based Model:**  
  Introduces a transformer architecture for 3D point cloud denoising in the diffusion-bridge framework.
- **Multi-Resolution Hash Encoding:**  
  Adopts InstantNGP-inspired encoding to capture multi-scale features efficiently, reducing memory usage compared to dense grids.
- **Robust Performance:**  
  Demonstrates state-of-the-art denoising on PCNet and PUNet datasets at 1% and 2% noise levels.
- **Flexible Design:**  
  Supports various architectural choices and refined input encoding techniques for further experimentation.

## Requirements
To begin, create a new environment with conda:

```bash
conda create -n p2pb python=3.10
conda activate p2pb
```

Install PyTorch and Torchvision first:
```bash
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```

Next, install [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and [TorchCluster](https://github.com/rusty1s/pytorch_cluster), followed by other dependencies and custom CUDA code:

```bash
sh install.sh
```

## Data Preparation

### Object Datasets (PU-Net and PC-Net)
1. Download the zip files from [ScoreDenoise](https://github.com/luost26/score-denoise).  
2. Extract them into `data/objects`:

   ```bash
   data/objects
   ├── examples
   ├── PCNet
   ├── PUNet
   ```

## Training
This project uses [wandb](https://wandb.ai/site) to track training progress. To log in:

```bash
wandb init
```

To disable logging:

```bash
wandb disabled
```

To train a model, edit the desired configuration file in `configs` and run:

```bash
python train.py --config <CONFIG FILE> \
                --save_dir <SAVE DIRECTORY> \
                --wandb_project <WANDB PROJECT NAME> \
                --wandb_entity <WANDB ENTITY NAME>
```

For more options:

```bash
python train.py --help
```

## Evaluation

### PU-Net and PC-Net
To evaluate on the PU-Net and PC-Net test sets:

```bash
python evaluate_objects.py --model_path <PATH_TO_PRETRAINED_MODEL> --dataset PUNet
python evaluate_objects.py --model_path <PATH_TO_PRETRAINED_MODEL> --dataset PCNet
```

Results and metrics are saved in `output_objects/<dataset>` by default. Use `--output_root` to specify a different output directory. For further details:

```bash
python evaluate_objects.py --help
```

## Denoise Objects
To denoise object data stored as an XYZ file, run:

```bash
python denoise_object.py \
    --data_path <PATH_TO_XYZ_FILE> \
    --save_path <OUTPUT_FILE> \
    --model_path <MODEL_PATH>
```

- **`--data_path`:** Path to your input `.xyz` file (point cloud).
- **`--save_path`:** Desired output path for the denoised point cloud.
- **`--model_path`:** Path to the pretrained model.

Before denoising, ensure that your input is correctly formatted. If your point clouds are in a different format (e.g., `.ply`, `.obj`), convert them to `.xyz` first.

## Acknowledgements
  This repository extends the P2P-Bridge project by 
    <a href="https://matvogel.github.io">Mathias Vogel</a>,
    <a href="https://scholar.google.com/citations?user=ml3laqEAAAAJ">Keisuke Tateno</a>,
    <a href="https://inf.ethz.ch/people/person-detail.pollefeys.html">Marc Pollefeys</a>,
    <a href="https://federicotombari.github.io/">Federico Tombari</a>,
    <a href="https://scholar.google.com/citations?user=eQ0om98AAAAJ">Marie-Julie Rakotosaona</a>, and
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a>.
    
We are grateful to the authors of the original P2P-Bridge for sharing their framework, enabling this further research on transformer-based point cloud denoising.

## Citation

If you use or build upon this work, please cite the following:

### P2P-Bridge
```bibtex
@inproceedings{vogel2024p2pbridgediffusionbridges3d,
      title={P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising}, 
      author={Mathias Vogel and Keisuke Tateno and Marc Pollefeys and Federico Tombari and Marie-Julie Rakotosaona and Francis Engelmann},
      year={2024},
      booktitle={European Conference on Computer Vision (ECCV)},
}
```

### InstantNGP
```bibtex
@article{mueller2022instant,
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2022},
    volume = {41},
    number = {4},
    month = jul,
    year = {2022},
    pages = {102:1--102:15},
    articleno = {102},
    numpages = {15},
    url = {https://doi.org/10.1145/3528223.3530127},
    doi = {10.1145/3528223.3530127},
    publisher = {ACM},
    address = {New York, NY, USA},
}
