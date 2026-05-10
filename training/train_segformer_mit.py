"""
SegFormer-like (U-Net decoder + MIT encoder) for wound healing.

Uses smp's mit_b* encoders. Grayscale input is replicated to 3 channels
since MIT encoders only support 3-channel input.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # type: ignore
    WoundDataset, build_train_transform, build_val_transform, make_collate,
)


class MitSegmentor(pl.LightningModule):
    def __init__(self, arch: str = "unet", encoder: str = "mit_b3",
                 lr: float = 1e-4, T_max: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(
            arch, encoder_name=encoder, encoder_weights="imagenet",
            in_channels=3, classes=1,
        )
        self.lr = lr; self.T_max = T_max
        self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
        self._step_outputs: dict[str, list] = {"train": [], "valid": [], "test": []}

    def forward(self, image):
        # image: (B, 1, H, W) uint8-encoded float; replicate to 3 channels
        x = image.float() / 255.0 - 0.5
        x = x.repeat(1, 3, 1, 1)
        return self.model(x)

    def _shared(self, batch, stage):
        image, mask = batch
        logits = self.forward(image)
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
        self.log_dict({f"{stage}_loss": loss, f"{stage}_iou": iou,
                       f"{stage}_dice": dice}, prog_bar=True, sync_dist=True)
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
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data/data/raw/dataset")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--arch", default="unet")
    p.add_argument("--encoder", default="mit_b3")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=6)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_float32_matmul_precision("medium")

    full_train = WoundDataset(args.data_root, "train", resize=None)
    test_ds = WoundDataset(args.data_root, "test", resize=None)
    val_size = max(1, int(np.ceil(args.val_ratio * len(full_train))))
    train_size = len(full_train) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    tl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=make_collate(build_train_transform(args.image_size)))
    vl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=make_collate(build_val_transform(args.image_size)))
    testl = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=make_collate(build_val_transform(args.image_size)))

    steps = max(1, len(tl) // max(1, args.accum))
    model = MitSegmentor(arch=args.arch, encoder=args.encoder, lr=args.lr,
                         T_max=args.epochs * steps)
    logger = TensorBoardLogger(args.out_dir, name="", version="tb")
    ckpt = ModelCheckpoint(dirpath=args.out_dir, filename="best",
                           monitor="valid_dice", mode="max", save_last=True)
    early = EarlyStopping(monitor="valid_dice", mode="max", patience=args.patience)
    trainer = pl.Trainer(
        max_epochs=args.epochs, precision="16-mixed", log_every_n_steps=1,
        logger=logger, callbacks=[ckpt, early], default_root_dir=args.out_dir,
        accumulate_grad_batches=args.accum,
    )
    trainer.fit(model, train_dataloaders=tl, val_dataloaders=vl)
    trainer.test(model, dataloaders=testl, ckpt_path="best")


if __name__ == "__main__":
    main()
