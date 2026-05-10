"""
Wound-healing scratch-assay segmentation — U-Net++ (ResNet-50) trainer.

Built for [INSTITUTION_GPU_HOST][redacted], RTX A5000 (GPU 1). Based on the existing
PyTorch-Lightning setup in wound-healing/backend/imgproc/unet/net.py,
with the following deltas:

  - Architecture: smp.UnetPlusPlus (vs. smp.Unet)
  - Mask binarization: (mask > 127) to remove JPEG compression artifacts on
    binary masks (values 1..7 near 0, 248..254 near 255 appear in this
    dataset).
  - Checkpoints + TensorBoard written to a timestamped run dir.
  - All hyperparameters read from a YAML config; CLI flag for dry-run.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
from dataclasses import dataclass

import albumentations as A
import imageio.v3 as iio
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchvision
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WoundDataset(torch.utils.data.Dataset):
    """Pairs images + masks in `<root>/<split>/{images,masks}/*.jpg`.

    Masks are binarized with `> 127` to drop JPEG compression artifacts.
    Images and masks are returned as uint8 torch tensors shaped (1, H, W);
    resizing and augmentation happen in the collate function.
    """

    def __init__(self, root: str | os.PathLike, split: str, resize: int | None):
        self.root = pathlib.Path(root) / split
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.names = sorted(os.listdir(self.img_dir))
        self.resize = resize

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = iio.imread(self.img_dir / name)
        mask = iio.imread(self.mask_dir / name)
        if img.ndim == 3:
            img = img[..., 0]  # force grayscale
        if mask.ndim == 3:
            mask = mask[..., 0]
        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W) uint8
        mask_t = torch.from_numpy((mask > 127).astype(np.uint8)).unsqueeze(0)
        if self.resize is not None:
            img_t = torchvision.transforms.functional.resize(
                img_t, self.resize, antialias=True
            )
            mask_t = torchvision.transforms.functional.resize(mask_t, self.resize)
        return img_t, mask_t


def _read_grayscale_ref(path: str) -> np.ndarray:
    arr = iio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def build_train_transform(
    size: int,
    mode: str = "default",
    hist_ref_glob: str | None = None,
) -> A.Compose:
    """Train aug pipeline. Mode selects shortcut-mitigation strategy:
      - default: paper-replica probabilistic CLAHE/brightness OneOf
      - always_clahe: apply CLAHE to every image (Fix A)
      - heavy_jitter: max brightness/contrast jitter + gamma every image (Fix B)
      - histogram_match: match each image to a random external reference (Fix C)
    """
    base_geom = [
        A.SafeRotate(limit=[-20, 20], p=0.3),
        A.RandomResizedCrop(size, size, scale=(0.45, 1.0), ratio=(0.85, 1.15), p=0.4),
        A.Resize(size, size, p=1),
    ]
    base_tail = [
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.GaussianBlur(p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        ], p=0.35),
    ]
    if mode == "default":
        photo = [A.OneOf([
            A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        ], p=0.6)]
    elif mode == "always_clahe":
        photo = [A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=1.0)]
    elif mode == "heavy_jitter":
        photo = [
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
            A.RandomGamma(gamma_limit=(50, 200), p=0.7),
        ]
    elif mode == "histogram_match":
        import glob as _glob
        if not hist_ref_glob:
            raise ValueError("histogram_match requires hist_ref_glob in config")
        refs = sorted(_glob.glob(hist_ref_glob))
        if not refs:
            raise ValueError(f"no images matched hist_ref_glob={hist_ref_glob}")
        photo = [A.HistogramMatching(
            reference_images=refs,
            blend_ratio=(0.5, 1.0),
            read_fn=_read_grayscale_ref,
            p=1.0,
        )]
    else:
        raise ValueError(f"unknown aug_mode: {mode}")
    return A.Compose(base_geom + photo + base_tail)


def build_val_transform(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size, p=1)])


def make_collate(transform: A.Compose):
    def collate(batch):
        imgs, masks = [], []
        for img_t, mask_t in batch:
            result = transform(
                image=img_t[0].numpy(), mask=mask_t[0].numpy()
            )
            imgs.append(torch.from_numpy(result["image"].copy()).unsqueeze(0))
            masks.append(torch.from_numpy(result["mask"].copy()).bool().unsqueeze(0))
        return torch.stack(imgs), torch.stack(masks)

    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SegmentationModel(pl.LightningModule):
    def __init__(self, arch: str, encoder: str, lr: float, T_max: int,
                 loss_name: str = "dice+bce",
                 aux_params: bool = False, aux_weight: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        aux = None
        if aux_params:
            # Aux classification head: regresses wound-pixel fraction [0, 1]
            # from global-pooled encoder features. Acts as deep-supervision
            # signal on the encoder — forces it to encode a meaningful global
            # summary of how much wound is present.
            aux = dict(classes=1, pooling="avg", dropout=0.2, activation=None)
        self.model = smp.create_model(
            arch,
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            aux_params=aux,
        )
        self.aux_params = aux_params
        self.aux_weight = aux_weight
        self.aux_loss_fn = torch.nn.BCEWithLogitsLoss() if aux_params else None
        self.lr = lr
        self.T_max = T_max
        if loss_name == "focal":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        elif loss_name == "dice":
            self.loss_fn = smp.losses.DiceLoss(
                smp.losses.BINARY_MODE, from_logits=True
            )
        elif loss_name == "dice+bce":
            self.loss_fn = _DiceBCE()
        elif loss_name == "dice+focal":
            self.loss_fn = _DiceFocal()
        else:
            raise ValueError(f"unknown loss: {loss_name}")
        self._step_outputs: dict[str, list] = {"train": [], "valid": [], "test": []}

    def forward(self, image):
        return self.model(image.float() / 255.0 - 0.5)

    def _shared(self, batch, stage):
        image, mask = batch
        assert image.shape[-1] % 32 == 0 and image.shape[-2] % 32 == 0, \
            f"H/W must be divisible by 32, got {image.shape}"
        out_model = self.forward(image)
        if self.aux_params:
            logits, aux_logits = out_model
            mask_f = mask.float()
            aux_target = mask_f.mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(1)
            seg_loss = self.loss_fn(logits, mask_f)
            aux_loss = self.aux_loss_fn(aux_logits, aux_target)
            loss = seg_loss + self.aux_weight * aux_loss
            self.log(f"{stage}_aux_loss", aux_loss.detach(), prog_bar=False,
                     on_step=False, on_epoch=True)
        else:
            logits = out_model
            loss = self.loss_fn(logits, mask.float())
        pred = (logits.sigmoid() > 0.5).long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, mask.long(), mode="binary")
        out = {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        self._step_outputs[stage].append({k: v.detach() for k, v in out.items()})
        return loss

    def _epoch_end(self, stage):
        outs = self._step_outputs[stage]
        if not outs:
            return
        tp = torch.cat([o["tp"] for o in outs])
        fp = torch.cat([o["fp"] for o in outs])
        fn = torch.cat([o["fn"] for o in outs])
        tn = torch.cat([o["tn"] for o in outs])
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        loss = torch.stack([o["loss"] for o in outs]).mean()
        self.log_dict({
            f"{stage}_loss": loss,
            f"{stage}_iou": iou,
            f"{stage}_dice": dice,
        }, prog_bar=True, sync_dist=True)
        outs.clear()

    def training_step(self, batch, _): return self._shared(batch, "train")
    def validation_step(self, batch, _): return self._shared(batch, "valid")
    def test_step(self, batch, _): return self._shared(batch, "test")
    def on_train_epoch_end(self): self._epoch_end("train")
    def on_validation_epoch_end(self): self._epoch_end("valid")
    def on_test_epoch_end(self): self._epoch_end("test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.T_max, eta_min=1e-6
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class _DiceBCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return 0.5 * self.dice(logits, target) + 0.5 * self.bce(logits, target)


class _DiceFocal(torch.nn.Module):
    """Dice + Focal: Focal attends to hard pixels (edges), Dice to overall IoU."""

    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.focal = smp.losses.FocalLoss(smp.losses.BINARY_MODE, alpha=0.5, gamma=2.0)

    def forward(self, logits, target):
        return 0.5 * self.dice(logits, target) + 0.5 * self.focal(logits, target)


def tta_predict(model, image: torch.Tensor) -> torch.Tensor:
    """Test-time augmentation: average logits across 8 flips/rots.

    model: the LightningModule (we call its forward(), which unpacks aux).
    image: (B, 1, H, W). Returns logits (B, 1, H, W).
    """
    views = []
    for hflip in (False, True):
        for vflip in (False, True):
            for rot90 in (0, 2):  # 0° and 180° (combined with flips ≈ all 8)
                x = image
                if hflip:
                    x = torch.flip(x, dims=[-1])
                if vflip:
                    x = torch.flip(x, dims=[-2])
                if rot90:
                    x = torch.rot90(x, k=rot90, dims=(-2, -1))
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                if rot90:
                    out = torch.rot90(out, k=-rot90, dims=(-2, -1))
                if vflip:
                    out = torch.flip(out, dims=[-2])
                if hflip:
                    out = torch.flip(out, dims=[-1])
                views.append(out)
    return torch.stack(views, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Config & entrypoint
# ---------------------------------------------------------------------------


@dataclass
class Config:
    data_root: str
    out_dir: str
    arch: str
    encoder: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    loss: str
    val_ratio: float
    num_workers: int
    seed: int
    patience: int
    precision: str
    swa: bool = False
    swa_start_epoch_ratio: float = 0.75
    tta: bool = False
    accumulate_grad_batches: int = 1
    aux_params: bool = False
    aux_weight: float = 0.2
    aug_mode: str = "default"
    hist_ref_glob: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None, help="override data_root")
    p.add_argument("--out-dir", default=None, help="override out_dir")
    p.add_argument("--dry-run", action="store_true",
                   help="1 epoch, 5 train + 2 val batches")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data_root = args.data_root
    if args.out_dir:
        cfg.out_dir = args.out_dir
    os.makedirs(cfg.out_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # Build datasets. We split train into train+val by val_ratio; test is held out.
    full_train = WoundDataset(cfg.data_root, "train", resize=None)
    test_ds = WoundDataset(cfg.data_root, "test", resize=None)
    val_size = max(1, int(np.ceil(cfg.val_ratio * len(full_train))))
    train_size = len(full_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=make_collate(build_train_transform(
            cfg.image_size, mode=cfg.aug_mode, hist_ref_glob=cfg.hist_ref_glob,
        )),
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=make_collate(build_val_transform(cfg.image_size)),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=make_collate(build_val_transform(cfg.image_size)),
    )

    steps_per_epoch = max(1, len(train_loader) // max(1, cfg.accumulate_grad_batches))
    model = SegmentationModel(
        arch=cfg.arch,
        encoder=cfg.encoder,
        lr=cfg.lr,
        T_max=cfg.epochs * steps_per_epoch,
        loss_name=cfg.loss,
        aux_params=cfg.aux_params,
        aux_weight=cfg.aux_weight,
    )

    logger = TensorBoardLogger(cfg.out_dir, name="", version="tb")
    ckpt = ModelCheckpoint(
        dirpath=cfg.out_dir,
        filename="best",
        monitor="valid_dice",
        mode="max",
        save_last=True,
    )
    early = EarlyStopping(monitor="valid_dice", mode="max", patience=cfg.patience)

    callbacks = [ckpt, early]
    if cfg.swa and not args.dry_run:
        swa_start = max(1, int(cfg.swa_start_epoch_ratio * cfg.epochs))
        callbacks.append(StochasticWeightAveraging(
            swa_lrs=cfg.lr * 0.1,
            swa_epoch_start=swa_start,
            annealing_epochs=5,
        ))
        print(f"[swa] enabled, start_epoch={swa_start}")

    trainer_kwargs = dict(
        max_epochs=1 if args.dry_run else cfg.epochs,
        precision=cfg.precision,
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )
    if args.dry_run:
        trainer_kwargs.update(limit_train_batches=5, limit_val_batches=2)

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if not args.dry_run:
        print("[test] running on held-out test set (standard) ...")
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")

        if cfg.tta:
            print("[test+tta] running with 8-way test-time augmentation ...")
            _test_with_tta(model, test_loader, ckpt_path=os.path.join(cfg.out_dir, "best.ckpt"))


def _test_with_tta(model: pl.LightningModule, loader, ckpt_path: str):
    """Manual TTA eval — reloads best checkpoint, averages 8 flip/rot views."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model = model.to("cuda").eval()
    tps = fps = fns = tns = None
    with torch.no_grad():
        for image, mask in loader:
            image = image.to("cuda", non_blocking=True)
            mask_d = mask.to("cuda", non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = tta_predict(model, image)
            pred = (logits.sigmoid() > 0.5).long()
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred, mask_d.long(), mode="binary"
            )
            tps = tp if tps is None else torch.cat([tps, tp])
            fps = fp if fps is None else torch.cat([fps, fp])
            fns = fn if fns is None else torch.cat([fns, fn])
            tns = tn if tns is None else torch.cat([tns, tn])
    iou = smp.metrics.iou_score(tps, fps, fns, tns, reduction="micro-imagewise").item()
    dice = smp.metrics.f1_score(tps, fps, fns, tns, reduction="micro-imagewise").item()
    print(f"[tta] test_dice={dice:.4f}  test_iou={iou:.4f}")


if __name__ == "__main__":
    main()
