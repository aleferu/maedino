#!/usr/bin/env python3

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


class DualViTSeg(nn.Module):
    def __init__(
        self,
        pretrained_mae=True,
        pretrained_dino=True,
        num_classes=3,
        img_size=320,
        patch_size=16,
        freeze_mae=False,
        freeze_dino=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # Initialize ViT Backbones via timm
        # We set num_classes=0 to get raw features instead of classification logits
        self.mae = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained_mae, img_size=img_size, num_classes=0)
        self.dino = timm.create_model("vit_base_patch16_224.dino", pretrained=pretrained_dino, img_size=img_size, num_classes=0)

        # Adapt from 3 channels (RGB) to 2 channels (T2, ADC)
        self._adapt_input_channels(self.mae)
        self._adapt_input_channels(self.dino)

        # Freeze backbones if requested
        if freeze_mae:
            for param in self.mae.parameters():
                param.requires_grad = False
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # CNN Decoder Pipeline
        # Both ViT bases output 768 dimensions. Fused = 768 + 768 = 1536
        embed_dim = self.mae.embed_dim
        fused_dim = embed_dim * 2

        # Upsampling from (H/16, W/16) back to (H, W) -> Requires 4 stages of 2x upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(fused_dim, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),  # Bottleneck to classes
        )

    def _adapt_input_channels(self, model):
        """Replaces the first projection layer to accept 2 channels instead of 3."""
        old_conv = model.patch_embed.proj
        new_conv = nn.Conv2d(
            2,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        # Weights for the two channels (T2 and ADC)
        with torch.no_grad():
            # Original weights averaged
            avg_weight = old_conv.weight.mean(dim=1, keepdim=True)

            # Apply it to both T2 and ADC
            new_conv.weight[:, 0:2, :, :] = avg_weight.repeat(1, 2, 1, 1)

        model.patch_embed.proj = new_conv

    def extract_spatial_features(self, model, x):
        """Passes input through ViT and restores the 2D spatial grid map."""
        # forward_features returns shape (Batch, Seq_Len, Embed_Dim)
        features = model.forward_features(x)

        # Remove CLS token(s). timm's num_prefix_tokens handles models with multiple prefix tokens.
        tokens = features[:, model.num_prefix_tokens:]

        # Reshape the flat token sequence back into a 2D grid
        # E.g., Seq_Len of 400 becomes 20x20 spatial grid
        B, N, C = tokens.shape
        grid_size = int(N ** 0.5)
        spatial_features = tokens.transpose(1, 2).reshape(B, C, grid_size, grid_size)

        return spatial_features

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width, Depth)
        B, C, H, W, D = x.shape

        # Slice: Convert 3D volume into batched 2D slices
        # Move Depth to Batch dimension -> (B*D, C, H, W)
        x_2d = x.permute(0, 4, 1, 2, 3).contiguous().reshape(B * D, C, H, W)

        # Extract spatial feature grids
        feat_mae = self.extract_spatial_features(self.mae, x_2d)
        feat_dino = self.extract_spatial_features(self.dino, x_2d)

        # Fuse representations (Concatenation)
        fused_features = torch.cat([feat_mae, feat_dino], dim=1)  # Shape: (B*D, 1536, H/16, W/16)

        # Decode to Masks
        masks_2d = self.decoder(fused_features)  # Shape: (B*D, Num_Classes, H, W)

        # Reconstruct 3D Volume
        # Reshape to (B, D, Num_Classes, H, W), then permute to (B, Num_Classes, H, W, D)
        out_3d = masks_2d.reshape(B, D, -1, H, W).permute(0, 2, 3, 4, 1).contiguous()

        return out_3d


def already_done(results_csv, exp_name, fold_number):
    """Returns True if this experiment/fold row already exists in results.csv."""
    if not results_csv.exists():
        return False
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == exp_name and int(row["fold"]) == fold_number:
                return True
    return False


def ensure_csv_header(path, header):
    """Writes the header row only if the file does not exist yet."""
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def main():
    parser = argparse.ArgumentParser(description="Train DualViTSeg with 5-fold CV")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["Exp_Both_Frozen", "Exp_MAE_Frozen_DINO_Un", "Exp_MAE_Un_DINO_Frozen", "Exp_Both_Unfrozen"],
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        metavar="FOLD",
        help="1-based fold indices to run (default: 1 2 3 4 5)",
    )
    args = parser.parse_args()

    DATASET_ROOT = Path("./Task05_Prostate")
    TRAIN_IMAGE_PATHS = [DATASET_ROOT / p for p in ["imagesTr/prostate_16.nii.gz", "imagesTr/prostate_04.nii.gz", "imagesTr/prostate_32.nii.gz", "imagesTr/prostate_20.nii.gz", "imagesTr/prostate_43.nii.gz", "imagesTr/prostate_18.nii.gz", "imagesTr/prostate_06.nii.gz", "imagesTr/prostate_14.nii.gz", "imagesTr/prostate_41.nii.gz", "imagesTr/prostate_34.nii.gz", "imagesTr/prostate_38.nii.gz", "imagesTr/prostate_10.nii.gz", "imagesTr/prostate_02.nii.gz", "imagesTr/prostate_24.nii.gz", "imagesTr/prostate_47.nii.gz", "imagesTr/prostate_28.nii.gz", "imagesTr/prostate_00.nii.gz", "imagesTr/prostate_42.nii.gz", "imagesTr/prostate_21.nii.gz", "imagesTr/prostate_17.nii.gz", "imagesTr/prostate_40.nii.gz", "imagesTr/prostate_31.nii.gz", "imagesTr/prostate_07.nii.gz", "imagesTr/prostate_35.nii.gz", "imagesTr/prostate_44.nii.gz", "imagesTr/prostate_39.nii.gz", "imagesTr/prostate_01.nii.gz", "imagesTr/prostate_13.nii.gz", "imagesTr/prostate_46.nii.gz", "imagesTr/prostate_25.nii.gz", "imagesTr/prostate_29.nii.gz", "imagesTr/prostate_37.nii.gz"]]
    TRAIN_LABEL_PATHS = [DATASET_ROOT / p for p in ["labelsTr/prostate_16.nii.gz", "labelsTr/prostate_04.nii.gz", "labelsTr/prostate_32.nii.gz", "labelsTr/prostate_20.nii.gz", "labelsTr/prostate_43.nii.gz", "labelsTr/prostate_18.nii.gz", "labelsTr/prostate_06.nii.gz", "labelsTr/prostate_14.nii.gz", "labelsTr/prostate_41.nii.gz", "labelsTr/prostate_34.nii.gz", "labelsTr/prostate_38.nii.gz", "labelsTr/prostate_10.nii.gz", "labelsTr/prostate_02.nii.gz", "labelsTr/prostate_24.nii.gz", "labelsTr/prostate_47.nii.gz", "labelsTr/prostate_28.nii.gz", "labelsTr/prostate_00.nii.gz", "labelsTr/prostate_42.nii.gz", "labelsTr/prostate_21.nii.gz", "labelsTr/prostate_17.nii.gz", "labelsTr/prostate_40.nii.gz", "labelsTr/prostate_31.nii.gz", "labelsTr/prostate_07.nii.gz", "labelsTr/prostate_35.nii.gz", "labelsTr/prostate_44.nii.gz", "labelsTr/prostate_39.nii.gz", "labelsTr/prostate_01.nii.gz", "labelsTr/prostate_13.nii.gz", "labelsTr/prostate_46.nii.gz", "labelsTr/prostate_25.nii.gz", "labelsTr/prostate_29.nii.gz", "labelsTr/prostate_37.nii.gz"]]

    train_transforms = Compose([
        # Load images
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),

        # Reorient all images to RAS convention (Nifti default)
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Resample everything to the median spacing we discovered
        Spacingd(keys=["image", "label"], pixdim=(0.625, 0.625, 3.6), mode=("bilinear", "nearest")),

        # Ensure all volumes are the same size
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(320, 320, 20)),

        # Z-score normalization for MRI, applied ONLY to the image
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Small rotations and translations
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(0.2, 0.2, 0.1),
            scale_range=(0.1, 0.1, 0.1),
        ),
        # Randomly flip the along the X axis (left/right symmetry)
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),

        # Adds noise
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.1),

        # To tensors
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        # Load images
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),

        # Reorient all images to RAS convention (Nifti default)
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Resample everything to the median spacing we discovered
        Spacingd(keys=["image", "label"], pixdim=(0.625, 0.625, 3.6), mode=("bilinear", "nearest")),

        # Ensure all volumes are the same size
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(320, 320, 20)),

        # Z-score normalization for MRI, applied ONLY to the image
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # To tensors
        ToTensord(keys=["image", "label"]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("DEVICE: %s", device)

    set_determinism(seed=42)

    data_dicts = np.array([
        {"image": img_path, "label": lbl_path}
        for img_path, lbl_path in zip(TRAIN_IMAGE_PATHS, TRAIN_LABEL_PATHS)
    ])

    experiments = [
        {"name": "Exp_Both_Frozen",        "freeze_mae": True,  "freeze_dino": True,  "batch_size": 5},
        {"name": "Exp_MAE_Frozen_DINO_Un", "freeze_mae": True,  "freeze_dino": False, "batch_size": 2},
        {"name": "Exp_MAE_Un_DINO_Frozen", "freeze_mae": False, "freeze_dino": True,  "batch_size": 2},
        {"name": "Exp_Both_Unfrozen",      "freeze_mae": False, "freeze_dino": False, "batch_size": 1},
    ]

    NUM_CLASSES = 3
    MAX_EPOCHS = 1000
    PATIENCE = 20
    LR_SCHEDULER_PATIENCE = PATIENCE // 4
    LR_SCHEDULER_GAMMA = 0.5
    NUM_WORKERS = 5
    LAMBDA_DICE = 5.0

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    weights = torch.tensor([1.0, 10.0, 10.0]).to(device)
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=weights,
        lambda_dice=LAMBDA_DICE,
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=False, reduction="mean_batch")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    # Output directories
    outputs_dir = Path("outputs")
    checkpoints_dir = Path("checkpoints")
    outputs_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    results_csv = outputs_dir / "results.csv"
    ensure_csv_header(results_csv, ["experiment", "fold", "dice_label1", "dice_label2", "mean_dice", "mean_iou", "hd95"])

    # Resolve the selected experiment config
    exp = next(e for e in experiments if e["name"] == args.experiment)
    exp_name = exp["name"]
    freeze_mae = exp["freeze_mae"]
    freeze_dino = exp["freeze_dino"]
    batch_size = exp["batch_size"]
    logging.info("\n=== Running Experiment: %s ===", exp_name)

    epoch_csv = outputs_dir / f"epochs_{exp_name}.csv"
    ensure_csv_header(epoch_csv, ["fold", "epoch", "train_loss", "val_mean_dice"])

    # Collect per-experiment results: {metric: [fold_values]}
    fold_dice1, fold_dice2, fold_mean_dice, fold_mean_iou, fold_hd95 = [], [], [], [], []

    # Run experiment! 5-fold CV
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_dicts)):
        fold_number = fold + 1

        # Skip folds not requested via --folds
        if fold_number not in args.folds:
            logging.info("%s | Fold %d/%d SKIPPED (not in --folds)", exp_name, fold_number, k_folds)
            continue

        # Skip folds already recorded in results.csv
        if already_done(results_csv, exp_name, fold_number):
            logging.info("%s | Fold %d/%d SKIPPED (already in results.csv)", exp_name, fold_number, k_folds)
            continue

        logging.info("%s | Fold %d/%d", exp_name, fold_number, k_folds)

        train_files = data_dicts[train_idx].tolist()
        val_files = data_dicts[val_idx].tolist()

        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        model = DualViTSeg(freeze_mae=freeze_mae, freeze_dino=freeze_dino).to(device)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-5,
        )
        scaler = torch.amp.GradScaler()
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=LR_SCHEDULER_GAMMA, patience=LR_SCHEDULER_PATIENCE)

        best_val_dice = -1.0
        epochs_no_improve = 0
        best_model_path = checkpoints_dir / f"best_model_{exp_name}_fold{fold_number}.pth"

        epoch_bar = tqdm(range(MAX_EPOCHS), desc=f"{exp_name} fold {fold_number}/{k_folds}", unit="epoch")
        for epoch in epoch_bar:
            # Training
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            # Validation
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(images)
                    pred = post_pred(outputs[0])
                    lbl = post_label(labels[0])
                    dice_metric(y_pred=pred.unsqueeze(0), y=lbl.unsqueeze(0))

            per_class_dice = dice_metric.aggregate()
            val_mean_dice = per_class_dice.mean().item()
            dice_metric.reset()

            avg_train_loss = epoch_loss / len(train_loader)
            scheduler.step(val_mean_dice)
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_bar.set_postfix(lr=current_lr, loss=avg_train_loss, val_dice=val_mean_dice)

            with open(epoch_csv, "a", newline="") as f:
                csv.writer(f).writerow([fold_number, epoch + 1, avg_train_loss, val_mean_dice])

            # Early stopping + best model save
            if val_mean_dice > best_val_dice:
                best_val_dice = val_mean_dice
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    tqdm.write(f"  Early stopping at epoch {epoch + 1}.")
                    break

        # Final evaluation on best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                pred = post_pred(outputs[0])
                lbl = post_label(labels[0])
                pred_b = pred.unsqueeze(0)
                lbl_b = lbl.unsqueeze(0)
                dice_metric(y_pred=pred_b, y=lbl_b)
                iou_metric(y_pred=pred_b, y=lbl_b)
                hd95_metric(y_pred=pred_b, y=lbl_b)

        per_class_dice_final = dice_metric.aggregate()
        mean_iou = iou_metric.aggregate().mean().item()
        hd95 = hd95_metric.aggregate().item()

        fold_dice1.append(per_class_dice_final[0].item())
        fold_dice2.append(per_class_dice_final[1].item())
        fold_mean_dice.append(per_class_dice_final.mean().item())
        fold_mean_iou.append(mean_iou)
        fold_hd95.append(hd95)

        logging.info(
            "  Fold %d results: Dice1=%f, Dice2=%f, MeanDice=%f, MeanIoU=%f, HD95=%f",
            fold_number, fold_dice1[-1], fold_dice2[-1], fold_mean_dice[-1], mean_iou, hd95,
        )

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([exp_name, fold_number, fold_dice1[-1], fold_dice2[-1], fold_mean_dice[-1], mean_iou, hd95])

        dice_metric.reset()
        iou_metric.reset()
        hd95_metric.reset()

    # Summary (only over folds we actually ran this session)
    if fold_dice1:
        logging.info("\n========== RESULTS SUMMARY ==========")
        logging.info("\n%s", exp_name)
        for metric_name, values in [
            ("dice_label1", fold_dice1),
            ("dice_label2", fold_dice2),
            ("mean_dice",   fold_mean_dice),
            ("mean_iou",    fold_mean_iou),
            ("hd95",        fold_hd95),
        ]:
            arr = np.array(values)
            logging.info("  %s: %f +/- %f", metric_name, arr.mean(), arr.std())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
