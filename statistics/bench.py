"""Benchmark inference latency + throughput for v1/v2/v3 checkpoints."""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import Config, SegmentationModel, tta_predict  # type: ignore


def make_model(cfg: Config):
    return SegmentationModel(
        arch=cfg.arch, encoder=cfg.encoder, lr=cfg.lr, T_max=1,
        loss_name=cfg.loss,
        aux_params=cfg.aux_params, aux_weight=cfg.aux_weight,
    )


def bench(name: str, cfg: Config, ckpt: str, tta: bool = False):
    m = make_model(cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"])
    m = m.to("cuda").eval()

    S = cfg.image_size
    x1 = torch.randn(1, 1, S, S, device="cuda")
    xB = torch.randn(cfg.batch_size, 1, S, S, device="cuda")

    def run(x, fn, warmup=5, reps=30):
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(warmup):
                fn(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn(x)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / reps

    fwd = (lambda x: tta_predict(m, x)) if tta else (
        lambda x: (m(x)[0] if isinstance(m(x), (tuple, list)) else m(x)))

    t_single = run(x1, fwd) * 1000  # ms
    t_batch = run(xB, fwd)
    ips_single = 1.0 / (t_single / 1000)
    ips_batch = cfg.batch_size / t_batch
    mem = torch.cuda.max_memory_allocated() / 1e9

    tag = f"{name}{'+tta' if tta else ''}"
    print(f"{tag:12s}  size={S:4d}  batch={cfg.batch_size:2d}  "
          f"single={t_single:6.1f} ms  ({ips_single:5.1f} img/s)  |  "
          f"batch={t_batch*1000/cfg.batch_size:5.1f} ms/img  ({ips_batch:5.1f} img/s)  |  "
          f"peak_vram={mem:4.1f} GB")

    del m
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", nargs="+", required=True,
                   help="name:config:ckpt triples")
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()
    for spec in args.specs:
        name, cfg_path, ckpt = spec.split(":")
        cfg = Config.from_yaml(cfg_path)
        bench(name, cfg, ckpt)
        if args.tta:
            bench(name, cfg, ckpt, tta=True)


if __name__ == "__main__":
    main()
