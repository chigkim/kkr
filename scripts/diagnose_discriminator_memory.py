#!/usr/bin/env python3
'''Focused MPD/MSD MPS memory diagnostic.

This reproduces the Stage 2 adversarial discriminator pattern without the full
Kokoro model. It is intended to isolate whether MPD, MSD, STFT, generator
backward, weight_norm, or upstream graph shapes cause MPS driver memory growth.
'''

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'StyleTTS2'))

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

import Modules.discriminators as discriminator_mod
from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator
from losses import (
    discriminator_loss,
    discriminator_TPRLS_loss,
    feature_loss,
    generator_loss,
    generator_TPRLS_loss,
)


def gb(n: int | float) -> float:
    return float(n) / 1024**3


def memory_line(tag: str, device: str) -> str:
    if device == 'mps' and torch.backends.mps.is_available():
        return (
            f'{tag}: cur={gb(torch.mps.current_allocated_memory()):.2f} '
            f'drv={gb(torch.mps.driver_allocated_memory()):.2f} '
            f'max={gb(torch.mps.recommended_max_memory()):.2f}'
        )
    if device == 'cuda' and torch.cuda.is_available():
        return (
            f'{tag}: cur={gb(torch.cuda.memory_allocated()):.2f} '
            f'reserved={gb(torch.cuda.memory_reserved()):.2f} '
            f'max={gb(torch.cuda.max_memory_allocated()):.2f}'
        )
    return f'{tag}: cpu'


def remove_weight_norm(module: torch.nn.Module) -> None:
    for m in module.modules():
        parametrizations = getattr(m, 'parametrizations', None)
        if not parametrizations:
            continue
        for name in list(parametrizations.keys()):
            parametrize.remove_parametrizations(m, name, leave_parametrized=True)


def patch_stft_on_cpu() -> None:
    original_stft = discriminator_mod.stft

    def cpu_stft(x, fft_size, hop_size, win_length, window):
        device = x.device
        y = original_stft(
            x.detach().to('cpu') if not x.requires_grad else x.to('cpu'),
            fft_size,
            hop_size,
            win_length,
            window.to('cpu') if torch.is_tensor(window) else window,
        )
        return y.to(device)

    discriminator_mod.stft = cpu_stft


def run_disc(module, y, y_hat, real_no_grad=False):
    return module(y, y_hat, real_no_grad=real_no_grad)


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def disc_loss_for(module, y, y_hat):
    out_device = y_hat.device
    module_device = _module_device(module)
    if module_device is not None:
        y = y.to(module_device)
        y_hat = y_hat.to(module_device)
    y_r, y_g, _, _ = run_disc(module, y, y_hat)
    loss, _, _ = discriminator_loss(y_r, y_g)
    loss = loss + discriminator_TPRLS_loss(y_r, y_g)
    return loss.mean().to(out_device)


def gen_loss_for(module, y, y_hat):
    out_device = y_hat.device
    module_device = _module_device(module)
    if module_device is not None:
        y = y.to(module_device)
        y_hat = y_hat.to(module_device)
    y_r, y_g, fmap_r, fmap_g = run_disc(module, y, y_hat, real_no_grad=True)
    loss_gen, _ = generator_loss(y_g)
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_rel = generator_TPRLS_loss(y_r, y_g)
    return (loss_gen + loss_fm + loss_rel).mean().to(out_device)


def make_fake(args, device, decoder, leaf):
    if args.graph_yrec:
        # Keep discriminator input fixed at segment_size while varying the
        # upstream graph width, matching the full-training hypothesis.
        extra = (args.step_mod and (args._step % args.step_mod) * args.step_stride) or 0
        length = args.segment_size + extra
        src = torch.randn(args.batch, 1, length, device=device)
        y = decoder(src)
        if y.size(-1) >= args.segment_size:
            y = y[..., : args.segment_size]
        else:
            y = F.pad(y, (0, args.segment_size - y.size(-1)))
        return y
    return leaf.expand(args.batch, -1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--mpd-device', default=None, help='override MPD device for split-device tests')
    parser.add_argument('--msd-device', default=None, help='override MSD device for split-device tests')
    parser.add_argument('--which', choices=['both', 'mpd', 'msd'], default='both')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--segment-size', type=int, default=16000)
    parser.add_argument('--d-only', action='store_true')
    parser.add_argument('--g-only', action='store_true')
    parser.add_argument('--graph-yrec', action='store_true', help='fake audio comes from a small conv with variable upstream length')
    parser.add_argument('--step-mod', type=int, default=7, help='number of upstream lengths to cycle through with --graph-yrec')
    parser.add_argument('--step-stride', type=int, default=512)
    parser.add_argument('--strip-weight-norm', action='store_true')
    parser.add_argument('--stft-on-cpu', action='store_true')
    parser.add_argument('--empty-cache-every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    args._step = 0

    assert not (args.d_only and args.g_only), 'Choose at most one of --d-only/--g-only'
    torch.manual_seed(args.seed)

    device = args.device
    mpd_device = args.mpd_device or device
    msd_device = args.msd_device or device
    if any(d == 'mps' for d in [device, mpd_device, msd_device]) and not torch.backends.mps.is_available():
        raise RuntimeError('MPS not available')
    if args.stft_on_cpu:
        patch_stft_on_cpu()

    modules = []
    mpd = msd = None
    if args.which in ('both', 'mpd'):
        mpd = MultiPeriodDiscriminator().to(mpd_device)
        modules.append(mpd)
    if args.which in ('both', 'msd'):
        msd = MultiResSpecDiscriminator().to(msd_device)
        modules.append(msd)

    if args.strip_weight_norm:
        for module in modules:
            remove_weight_norm(module)

    disc_params = [p for module in modules for p in module.parameters()]
    dopt = torch.optim.AdamW(disc_params, lr=1e-5) if disc_params else None

    decoder = None
    gopt = None
    leaf = None
    if args.graph_yrec:
        decoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, 5, padding=2),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(16, 1, 5, padding=2),
        ).to(device)
        gopt = torch.optim.AdamW(decoder.parameters(), lr=1e-5)
    else:
        leaf = torch.nn.Parameter(torch.randn(1, 1, args.segment_size, device=device) * 0.01)
        gopt = torch.optim.AdamW([leaf], lr=1e-5)

    print(
        f'which={args.which} device={device} mpd_device={mpd_device} msd_device={msd_device} steps={args.steps} batch={args.batch} '
        f'segment={args.segment_size} graph_yrec={args.graph_yrec} '
        f'd_only={args.d_only} g_only={args.g_only} strip_weight_norm={args.strip_weight_norm} '
        f'stft_on_cpu={args.stft_on_cpu}',
        flush=True,
    )
    print(memory_line('start', device), flush=True)

    for step in range(1, args.steps + 1):
        args._step = step
        wav = torch.randn(args.batch, 1, args.segment_size, device=device)
        y_hat = make_fake(args, device, decoder, leaf)

        d_loss = torch.tensor(0.0, device=device)
        g_loss = torch.tensor(0.0, device=device)

        if not args.g_only:
            dopt.zero_grad(set_to_none=True)
            if mpd is not None:
                d_loss = d_loss + disc_loss_for(mpd, wav.detach(), y_hat.detach())
            if msd is not None:
                d_loss = d_loss + disc_loss_for(msd, wav.detach(), y_hat.detach())
            d_loss.backward()
            dopt.step()

        if not args.d_only:
            gopt.zero_grad(set_to_none=True)
            # Intentionally leave discriminator params requires_grad=True, matching
            # train_second.py's generator adversarial path.
            if mpd is not None:
                g_loss = g_loss + gen_loss_for(mpd, wav.detach(), y_hat)
            if msd is not None:
                g_loss = g_loss + gen_loss_for(msd, wav.detach(), y_hat)
            g_loss.backward()
            gopt.step()

        if args.empty_cache_every and device == 'mps' and step % args.empty_cache_every == 0:
            gc.collect()
            torch.mps.empty_cache()

        if step == 1 or step % args.log_interval == 0:
            print(
                f'step={step} d={float(d_loss.detach().cpu()):.4f} '
                f'g={float(g_loss.detach().cpu()):.4f} {memory_line("", device)}',
                flush=True,
            )

    print(memory_line('end', device), flush=True)


if __name__ == '__main__':
    main()
