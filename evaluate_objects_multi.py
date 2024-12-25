import argparse
import os
from typing import Any, Dict, Generator, List

import numpy as np
import omegaconf
import pytorch3d.ops
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from models.evaluation import Evaluator, farthest_point_sampling
from models.model_loader import load_diffusion
from models.train_utils import set_seed
from utils.utils import NormalizeUnitSphere, write_array_to_xyz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="./data/objects/examples/", help="Path to the room point cloud."
    )
    parser.add_argument("--output_root", type=str, default="./output_objects", help="Output root directory.")
    parser.add_argument("--dataset_root", type=str, default="./data/objects/", help="Path to the dataset root.")
    parser.add_argument(
        "--model_path", type=str, default="./pretrained/PVDS_PUNet/latest.pth", help="Path to the model."
    )
    parser.add_argument("--dataset", type=str, default="PUNet", help="Dataset name.", choices=["PUNet", "PCNet"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=3, help="Patch oversampling factor.")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model for prediction.")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate steps.")
    parser.add_argument("--gpu", type=str, default="cuda", help="GPU(s) to use. Example: 'cuda:0,1'")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps for the diffusion.")
    parser.add_argument("--distribution_type", default="none")
    args = parser.parse_args()

    # Load config from checkpoint
    cfg_path = os.path.join(os.path.dirname(args.model_path), "opt.yaml")
    cfg = omegaconf.OmegaConf.load(cfg_path)

    # Merge with args
    cfg = omegaconf.OmegaConf.merge(cfg, omegaconf.OmegaConf.create(vars(args)))

    # Set some additional parameters
    cfg.restart = False
    cfg.local_rank = 0
    return cfg


def input_iter(input_dir: str) -> Generator[Any, Any, Any]:
    """
    Iterate over the input directory and yield the point cloud data.

    Args:
        input_dir (str): The input directory.

    Yields:
        Any: The point cloud data.
    """

    for fn in os.listdir(input_dir):
        if not fn.endswith(".xyz"):
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {"pcl_noisy": pcl_noisy, "name": fn[:-4], "center": center, "scale": scale}


@torch.no_grad()
def patch_based_denoise(
    model: torch.nn.Module,
    pcl_noisy: torch.Tensor,
    patch_size: int,
    seed_k: int = 3,
    cfg: DictConfig = None,
    save_intermediate: bool = False,
    batch_size: int = 128,  # Adjust batch_size as needed
):
    """
    Perform patch-based denoising on a noisy point cloud.

    Args:
        model (torch.nn.Module): The model to use for denoising.
        pcl_noisy (torch.Tensor): The noisy point cloud.
        patch_size (int): The size of the patches.
        seed_k (int): The oversampling factor for the seed points.
        cfg (DictConfig): The configuration dictionary.
        save_intermediate (bool): Whether to save intermediate steps.
        batch_size (int): Number of patches to process in a batch.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: The denoised point cloud and the intermediate steps.
    """
    assert pcl_noisy.dim() == 2, "The shape of input point cloud must be (N, 3)."
    N, d = pcl_noisy.size()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcl_noisy = pcl_noisy.unsqueeze(0).to(device)  # (1, N, 3)

    # Farthest point sampling to select seed points
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]  # (num_patches, K, 3)

    model.eval()

    # Center and scale the patches
    centers = patches.mean(dim=1, keepdim=True)
    patches = patches - centers
    scale = torch.max(torch.norm(patches, dim=-1, keepdim=True), dim=1, keepdim=True)[0]
    patches = patches / scale

    num_patches = patches.size(0)
    denoised_patches = []
    patches_steps = [] if save_intermediate else None

    # Process patches in batches to save memory
    for i in range(0, num_patches, batch_size):
        batch_patches = patches[i:i + batch_size].transpose(1, 2).to(device)  # (B, 3, K)

        # Perform sampling with the model
        out = model.sample(
            x_start=batch_patches,
            use_ema=cfg.use_ema,
            steps=cfg.steps,
            log_count=cfg.steps,
            verbose=False
        )

        # Collect denoised patches
        batch_denoised = out["x_pred"].transpose(1, 2).cpu()  # (B, K, 3)
        denoised_patches.append(batch_denoised)

        if save_intermediate:
            # Collect intermediate steps
            batch_steps = out["x_chain"].transpose(-2, -1).cpu()  # (T, B, K, 3)
            patches_steps.append(batch_steps)

        # Free up GPU memory
        del batch_patches, out
        torch.cuda.empty_cache()

    # Concatenate all denoised patches
    denoised_patches = torch.cat(denoised_patches, dim=0)  # (num_patches, K, 3)
    denoised_patches = denoised_patches * scale.cpu() + centers.cpu()

    if save_intermediate and patches_steps:
        # Concatenate all intermediate steps
        patches_steps = torch.cat(patches_steps, dim=1)  # (T, num_patches, K, 3)
        T, num_patches, K, d = patches_steps.size()
        patches_steps = patches_steps.view(T, num_patches * K, d)
        pcl_steps_denoised, _ = farthest_point_sampling(patches_steps, N)
    else:
        patches_steps = None

    # Aggregate denoised patches into a single point cloud
    pcl_denoised, _ = farthest_point_sampling(denoised_patches.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0].cpu().squeeze()

    return pcl_denoised, patches_steps


@torch.no_grad()
def sample(
    cfg: DictConfig,
    resolutions: List[int] = [50000],
    noises: List[float] = [0.01, 0.02, 0.03],
    save_title: str = "P2P-Bridge",
) -> None:
    """
    Sample from the model with DataParallel using multiple GPUs.

    Args:
        cfg (DictConfig): The configuration dictionary.
        resolutions (List[int]): The resolutions to sample.
        noises (List[float]): The noise levels to sample.
        save_title (str): The title to save the results.
    """
    set_seed(cfg)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model, _ = load_diffusion(cfg)
    model = model.to(device)

    # Set the model to evaluation mode before wrapping with DataParallel
    model.eval()

    # Utilize multiple GPUs if available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)

    out_root = os.path.join(cfg.output_root, cfg.dataset)

    if cfg.use_ema:
        save_title += "_ema"
    save_title += f"_steps_{cfg.steps}"

    for res in resolutions:
        for noise in noises:
            input_dir = os.path.join(cfg.data_path, f"{cfg.dataset}_{res}_poisson_{noise}")
            output_dir = os.path.join(out_root, f"{save_title}_{res}_{noise}")

            for data in input_iter(input_dir):
                logger.info(f"Processing {data['name']}")
                pcl_noisy = data["pcl_noisy"].to(device)

                with torch.no_grad():
                    pcl_denoised, pcl_steps_denoised = patch_based_denoise(
                        model=model,
                        pcl_noisy=pcl_noisy,
                        patch_size=2048,
                        seed_k=cfg.k,
                        cfg=cfg,
                        save_intermediate=cfg.save_intermediate,
                        batch_size=128,  # Adjust as needed
                    )

                pcl_denoised = pcl_denoised * data["scale"] + data["center"]

                save_path = os.path.join(output_dir, "pcl", f"{data['name']}.xyz")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                write_array_to_xyz(save_path, pcl_denoised.numpy())

                if cfg.save_intermediate and pcl_steps_denoised is not None:
                    for step, item in enumerate(pcl_steps_denoised):
                        pcl_step = item.cpu()
                        pcl_step = pcl_step * data["scale"] + data["center"]
                        out_path = os.path.join(output_dir, "steps", data["name"], f"{data['name']}_{step}.xyz")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        write_array_to_xyz(out_path, pcl_step.numpy())

                # Free up GPU memory after processing each cloud
                del pcl_noisy, pcl_denoised
                if cfg.save_intermediate and pcl_steps_denoised is not None:
                    del pcl_steps_denoised
                torch.cuda.empty_cache()

            evaluator = Evaluator(
                output_pcl_dir=os.path.join(output_dir, "pcl"),
                dataset_root=cfg.dataset_root,
                dataset=cfg.dataset,
                summary_dir=output_dir,
                experiment_name=save_title,
                device=device,
                res_gts=f"{res}_poisson",
            )
            evaluator.run()


def main():
    opt = parse_args()
    sample(opt)


if __name__ == "__main__":
    main()
