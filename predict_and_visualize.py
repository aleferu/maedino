#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism
from sklearn.model_selection import KFold

from both import DualViTSeg
from onlydino import DINOViTSeg
from onlymae import MAEViTSeg


DATASET_ROOT = Path("./Task05_Prostate")
DATASET_JSON = DATASET_ROOT / "dataset.json"

TRAIN_IMAGE_PATHS = [DATASET_ROOT / p for p in ["imagesTr/prostate_16.nii.gz", "imagesTr/prostate_04.nii.gz", "imagesTr/prostate_32.nii.gz", "imagesTr/prostate_20.nii.gz", "imagesTr/prostate_43.nii.gz", "imagesTr/prostate_18.nii.gz", "imagesTr/prostate_06.nii.gz", "imagesTr/prostate_14.nii.gz", "imagesTr/prostate_41.nii.gz", "imagesTr/prostate_34.nii.gz", "imagesTr/prostate_38.nii.gz", "imagesTr/prostate_10.nii.gz", "imagesTr/prostate_02.nii.gz", "imagesTr/prostate_24.nii.gz", "imagesTr/prostate_47.nii.gz", "imagesTr/prostate_28.nii.gz", "imagesTr/prostate_00.nii.gz", "imagesTr/prostate_42.nii.gz", "imagesTr/prostate_21.nii.gz", "imagesTr/prostate_17.nii.gz", "imagesTr/prostate_40.nii.gz", "imagesTr/prostate_31.nii.gz", "imagesTr/prostate_07.nii.gz", "imagesTr/prostate_35.nii.gz", "imagesTr/prostate_44.nii.gz", "imagesTr/prostate_39.nii.gz", "imagesTr/prostate_01.nii.gz", "imagesTr/prostate_13.nii.gz", "imagesTr/prostate_46.nii.gz", "imagesTr/prostate_25.nii.gz", "imagesTr/prostate_29.nii.gz", "imagesTr/prostate_37.nii.gz"]]
TRAIN_LABEL_PATHS = [DATASET_ROOT / p for p in ["labelsTr/prostate_16.nii.gz", "labelsTr/prostate_04.nii.gz", "labelsTr/prostate_32.nii.gz", "labelsTr/prostate_20.nii.gz", "labelsTr/prostate_43.nii.gz", "labelsTr/prostate_18.nii.gz", "labelsTr/prostate_06.nii.gz", "labelsTr/prostate_14.nii.gz", "labelsTr/prostate_41.nii.gz", "labelsTr/prostate_34.nii.gz", "labelsTr/prostate_38.nii.gz", "labelsTr/prostate_10.nii.gz", "labelsTr/prostate_02.nii.gz", "labelsTr/prostate_24.nii.gz", "labelsTr/prostate_47.nii.gz", "labelsTr/prostate_28.nii.gz", "labelsTr/prostate_00.nii.gz", "labelsTr/prostate_42.nii.gz", "labelsTr/prostate_21.nii.gz", "labelsTr/prostate_17.nii.gz", "labelsTr/prostate_40.nii.gz", "labelsTr/prostate_31.nii.gz", "labelsTr/prostate_07.nii.gz", "labelsTr/prostate_35.nii.gz", "labelsTr/prostate_44.nii.gz", "labelsTr/prostate_39.nii.gz", "labelsTr/prostate_01.nii.gz", "labelsTr/prostate_13.nii.gz", "labelsTr/prostate_46.nii.gz", "labelsTr/prostate_25.nii.gz", "labelsTr/prostate_29.nii.gz", "labelsTr/prostate_37.nii.gz"]]

EXPERIMENT_MODEL_MAP = {
    "Exp_Both_Frozen":        "dual",
    "Exp_MAE_Frozen_DINO_Un": "dual",
    "Exp_MAE_Un_DINO_Frozen": "dual",
    "Exp_Both_Unfrozen":      "dual",
    "Exp_MAE_Frozen":         "mae",
    "Exp_MAE_Unfrozen":       "mae",
    "Exp_Dino_Frozen":        "dino",
    "Exp_Dino_Unfrozen":      "dino",
}

NUM_CLASSES = 3
K_FOLDS = 5

LABEL_COLORS = np.array([
    [0,   0,   0],
    [220, 50,  50],
    [50,  100, 220],
], dtype=np.uint8)


def build_val_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.625, 0.625, 3.6), mode=("bilinear", "nearest")),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(320, 320, 20)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


def build_test_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.625, 0.625, 3.6), mode="bilinear"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(320, 320, 20)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def get_val_files(fold_number: int) -> list[dict]:
    """Returns the validation file dicts for the given 1-based fold."""
    data_dicts = np.array([
        {"image": img, "label": lbl}
        for img, lbl in zip(TRAIN_IMAGE_PATHS, TRAIN_LABEL_PATHS)
    ])
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    for fold_idx, (_, val_idx) in enumerate(kf.split(data_dicts)):
        if fold_idx + 1 == fold_number:
            return data_dicts[val_idx].tolist()
    raise ValueError("Fold %d not found" % fold_number)


def get_test_paths() -> list[Path]:
    """Returns test image paths from dataset.json."""
    with open(DATASET_JSON) as f:
        meta = json.load(f)
    return [DATASET_ROOT / p.lstrip("./") for p in meta["test"]]


def build_model(exp_name: str) -> nn.Module:
    tag = EXPERIMENT_MODEL_MAP[exp_name]
    if tag == "dual":
        return DualViTSeg()
    if tag == "mae":
        return MAEViTSeg()
    return DINOViTSeg()


def load_model(exp_name: str, fold_number: int, checkpoints_dir: Path, device: torch.device) -> nn.Module:
    model = build_model(exp_name).to(device)
    ckpt_path = checkpoints_dir / f"best_model_{exp_name}_fold{fold_number}.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Returns argmax prediction as (H, W, D) int array."""
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            logits = model(image_tensor.unsqueeze(0).to(device))
    return logits[0].argmax(dim=0).cpu().numpy()


def label_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Converts an integer label map (H, W) to an RGB image."""
    return LABEL_COLORS[mask.astype(int)]


def pick_slices(pred: np.ndarray, n: int) -> list[int]:
    """Picks n evenly-spaced slice indices that contain at least some foreground."""
    D = pred.shape[2]
    foreground = [z for z in range(D) if pred[:, :, z].max() > 0]
    pool = foreground if foreground else list(range(D))
    indices = [pool[int(i * len(pool) / n)] for i in range(n)]
    return indices


def save_val_figure(image: np.ndarray, pred: np.ndarray, label: np.ndarray,
                    slice_idx: int, volume_name: str, exp_name: str,
                    fold_number: int, out_path: Path) -> None:
    """Three-panel figure: T2 slice | prediction | ground truth."""
    t2 = image[0, :, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]
    label_slice = label[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(t2, cmap="gray", origin="lower")
    axes[0].set_title(f"T2 - {volume_name} slice {slice_idx}")
    axes[1].imshow(t2, cmap="gray", origin="lower")
    axes[1].imshow(label_to_rgb(pred_slice), alpha=0.5, origin="lower")
    axes[1].set_title(f"Prediction - {exp_name} fold {fold_number}")
    axes[2].imshow(t2, cmap="gray", origin="lower")
    axes[2].imshow(label_to_rgb(label_slice), alpha=0.5, origin="lower")
    axes[2].set_title("Ground truth")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_test_figure(image: np.ndarray, pred: np.ndarray,
                     slice_idx: int, volume_name: str, exp_name: str,
                     fold_number: int, out_path: Path) -> None:
    """Two-panel figure: T2 slice | prediction (no ground truth for test)."""
    t2 = image[0, :, :, slice_idx]
    pred_slice = pred[:, :, slice_idx]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(t2, cmap="gray", origin="lower")
    axes[0].set_title(f"T2 - {volume_name} slice {slice_idx}")
    axes[1].imshow(t2, cmap="gray", origin="lower")
    axes[1].imshow(label_to_rgb(pred_slice), alpha=0.5, origin="lower")
    axes[1].set_title(f"Prediction - {exp_name} fold {fold_number}")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise model predictions on validation and test volumes")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENT_MODEL_MAP.keys()), help="Experiment name")
    parser.add_argument("--fold", required=True, type=int, choices=list(range(1, K_FOLDS + 1)), help="1-based fold number")
    parser.add_argument("--n-slices", type=int, default=5, metavar="N", help="Number of slice images to save per volume (default: 5)")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Directory containing .pth checkpoints")
    parser.add_argument("--out-dir", type=str, default="predictions", help="Output directory for figures (default: predictions)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for validation volume selection (default: 0)")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    out_dir = Path(args.out_dir) / f"{args.experiment}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(seed=42)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.experiment, args.fold, checkpoints_dir, device)

    val_files = get_val_files(args.fold)
    chosen_val = random.choice(val_files)
    volume_name = Path(chosen_val["image"]).stem.replace(".nii", "")
    print(f"Validation volume: {volume_name}")

    val_transforms = build_val_transforms()
    val_sample = val_transforms(chosen_val)
    image_tensor = val_sample["image"]
    label_tensor = val_sample["label"]

    pred = predict(model, image_tensor, device)
    image_np = image_tensor.numpy()
    label_np = label_tensor[0].numpy().astype(int)

    slice_indices = pick_slices(pred, args.n_slices)
    for z in slice_indices:
        out_path = out_dir / f"val_{volume_name}_slice{z:02d}.png"
        save_val_figure(image_np, pred, label_np, z, volume_name, args.experiment, args.fold, out_path)
        print(f"Saved {out_path}")

    test_paths = get_test_paths()
    chosen_test = random.choice(test_paths)
    test_name = chosen_test.stem.replace(".nii", "")
    print(f"Test volume: {test_name}")

    test_transforms = build_test_transforms()
    test_sample = test_transforms({"image": chosen_test})
    test_tensor = test_sample["image"]

    test_pred = predict(model, test_tensor, device)
    test_np = test_tensor.numpy()

    test_slice_indices = pick_slices(test_pred, args.n_slices)
    for z in test_slice_indices:
        out_path = out_dir / f"test_{test_name}_slice{z:02d}.png"
        save_test_figure(test_np, test_pred, z, test_name, args.experiment, args.fold, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
