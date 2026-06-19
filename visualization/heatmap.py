import logging
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.basic_config import get_args
from data.cub_loader import CONCEPT_SEMANTICS, SELECTED_CONCEPTS, CUBDataset
from models.construction import construct_model
from utils import logging_config


def visualize_and_save_heatmaps(
        x,
        c_ground_truth,
        c_pred,
        heatmap,
        output_dir="output_images",
        concept_set=None,
):
    os.makedirs(output_dir, exist_ok=True)

    image = x.permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image * 0.5 + 0.5, 0, 1)
    image_bgr = cv2.cvtColor(np.uint8(255 * image), cv2.COLOR_RGB2BGR)

    plt.imsave(os.path.join(output_dir, "original_image.png"), image)

    for concept_idx in range(heatmap.shape[0]):
        hm = heatmap[concept_idx].detach().cpu()
        if hm.max().item() < 1e-4:
            continue

        hm = hm.unsqueeze(0).unsqueeze(0)
        hm = F.interpolate(
            hm,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
        overlay_bgr = cv2.addWeighted(image_bgr, 0.6, heatmap_bgr, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        concept_name = concept_set[concept_idx] if concept_set is not None else f"concept_{concept_idx}"
        safe_name = concept_name.replace(":", "_").replace("/", "_")
        gt_val = c_ground_truth[concept_idx].item()
        pred_val = c_pred[concept_idx].item()

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_rgb)
        plt.axis("off")
        plt.title(f"{concept_name}\nGT:[{gt_val:.0f}] Pred:[{pred_val:.3f}]", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
        plt.close()


def main():
    args = get_args()
    if args.checkpoint_dir is None:
        raise ValueError("Please pass --checkpoint_dir with the directory containing FixCBM.pt.")

    config_path = os.path.join(PROJECT_ROOT, "configs", f"{args.dataset}.yaml")
    with open(config_path, "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)

    target_run = experiment_config["runs"][0]
    run_config = dict(experiment_config)
    run_config.update(target_run)
    run_config["c_extractor_arch"] = args.image_encoder

    current_time = time.strftime("%H-%M", time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_heatmap_{current_time}")
    logging_config(save_dir)

    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    model = construct_model(
        n_concepts=112,
        n_tasks=200,
        config=run_config,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
    ])

    root_dir = os.path.join(PROJECT_ROOT, "data", "CUB_200_2011")
    test_data_path = os.path.join(root_dir, "class_attr_data_10", "test.pkl")
    dataset = CUBDataset(
        pkl_file_paths=[test_data_path],
        image_dir="images",
        labeled_ratio=1.0,
        training=False,
        transform=transform,
        root_dir=root_dir,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    concept_set = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]

    heatmap_dir = os.path.join(save_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    logging.info(f"Saving heatmaps to: {heatmap_dir}")

    for batch_idx, batch in enumerate(loader):
        x, _, c, _ = batch
        x = x.to(device)
        c = c.to(device)

        with torch.no_grad():
            outputs = model._forward(x, train=False)
            c_pred = outputs[0]
            heatmap_tensor = model.plot_heatmap(x)

        if heatmap_tensor is None:
            raise RuntimeError("The loaded model does not provide heatmaps.")

        for sample_idx in range(x.shape[0]):
            sample_dir = os.path.join(heatmap_dir, f"batch_{batch_idx}_img_{sample_idx}")
            visualize_and_save_heatmaps(
                x=x[sample_idx].cpu(),
                c_ground_truth=c[sample_idx].cpu(),
                c_pred=c_pred[sample_idx].cpu(),
                heatmap=heatmap_tensor[sample_idx].cpu(),
                output_dir=sample_dir,
                concept_set=concept_set,
            )
        break

    logging.info("Heatmap generation finished.")


if __name__ == "__main__":
    main()
