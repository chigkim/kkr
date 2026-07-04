#!/usr/bin/env python
"""Reproduce MPD/MSD MPS driver-memory growth in isolation.

Runs each discriminator (and each sub-component) on MPS with a FIXED input
length for many fwd/bwd iterations and reports MPS current/driver memory so we
can attribute growth to a specific op (torch.stft, reflect pad, window.to,
weight_norm/spectral_norm, conv2d).

Usage:
    python scripts/diagnose_discriminator_mps_memory.py --target msd --iters 200
    python scripts/diagnose_discriminator_mps_memory.py --target mpd --iters 200
    python scripts/diagnose_discriminator_mps_memory.py --target stft_only --iters 200
    python scripts/diagnose_discriminator_mps_memory.py --target all --segment 24000

Flags let us isolate fixes:
    --empty-cache     call torch.mps.empty_cache() each iter (is growth just cache?)
    --no-backward     forward only (is the leak in autograd?)
    --cache-window    monkeypatch SpecDiscriminator to reuse a buffered window
    --stft-cpu        run MSD stft on CPU, convs on MPS (proposed fix)
"""
import argparse
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "StyleTTS2"))

from Modules.discriminators import (  # noqa: E402
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    SpecDiscriminator,
    stft,
)


def gb(x):
    return float(x) / 1024 ** 3


def mem_line(dev):
    if dev == "mps":
        return "cur=%.3f drv=%.3f" % (
            gb(torch.mps.current_allocated_memory()),
            gb(torch.mps.driver_allocated_memory()),
        )
    return "n/a"


def patch_cache_window(disc):
    """Register the window as a buffer and reuse it (no per-forward .to())."""
    for m in disc.modules():
        if isinstance(m, SpecDiscriminator):
            if "window" in m._parameters or hasattr(m, "_buffers") and "window" in m._buffers:
                continue
            w = m.window
            del m.window
            m.register_buffer("window", w)

            def make_fwd(mod):
                def fwd(y):
                    fmap = []
                    y = y.squeeze(1)
                    y = stft(y, mod.fft_size, mod.shift_size, mod.win_length, mod.window)
                    y = y.unsqueeze(1)
                    for d in mod.discriminators:
                        y = F.leaky_relu(d(y), 0.1)
                        fmap.append(y)
                    y = mod.out(y)
                    fmap.append(y)
                    return torch.flatten(y, 1, -1), fmap

                return fwd

            m.forward = make_fwd(m)


def run(target, iters, segment, dev, backward, empty_cache, cache_window):
    torch.manual_seed(0)

    if target == "mpd":
        disc = MultiPeriodDiscriminator().to(dev)
    elif target == "msd":
        disc = MultiResSpecDiscriminator().to(dev)
        if cache_window:
            patch_cache_window(disc)
    elif target == "stft_only":
        win = torch.hann_window(600).to(dev)
        print("stft_only: fixed-length torch.stft + abs on", dev)
    else:
        raise ValueError(target)

    print(f"[{target}] start  {mem_line(dev)}")
    for i in range(iters):
        y = torch.randn(2, 1, segment, device=dev, requires_grad=backward)
        y_hat = torch.randn(2, 1, segment, device=dev, requires_grad=backward)

        if target == "stft_only":
            out = stft(y.squeeze(1), 1024, 120, 600, win)
            loss = out.abs().mean()
        else:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
            loss = sum(o.mean() for o in y_d_rs) + sum(o.mean() for o in y_d_gs)

        if backward:
            loss.backward()
        if empty_cache:
            torch.mps.empty_cache()
        if i % 25 == 0 or i == iters - 1:
            print(f"[{target}] iter {i:4d} loss={loss.item():.4f}  {mem_line(dev)}")
    print(f"[{target}] end    {mem_line(dev)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="all",
                    choices=["mpd", "msd", "stft_only", "all"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--segment", type=int, default=24000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--no-backward", dest="backward", action="store_false")
    ap.add_argument("--empty-cache", action="store_true")
    ap.add_argument("--cache-window", action="store_true")
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        sys.exit("MPS not available")

    targets = ["stft_only", "msd", "mpd"] if args.target == "all" else [args.target]
    for t in targets:
        run(t, args.iters, args.segment, args.device,
            args.backward, args.empty_cache, args.cache_window)
        print("-" * 60)


if __name__ == "__main__":
    main()
