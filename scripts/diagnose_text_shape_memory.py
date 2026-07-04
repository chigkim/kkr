#!/usr/bin/env python3
'''Exercise Stage 2 variable text/attention backward path and report MPS driver memory.'''

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.90')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.80')

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLETTS2_ROOT = REPO_ROOT / 'StyleTTS2'
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STYLETTS2_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLETTS2_ROOT))

import torch
import numpy as np
import torch.nn.functional as F
import yaml
from munch import Munch

import train_second as ts
from losses import MultiResolutionSTFTLoss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
from Utils.PLBERT.util import load_plbert
from utils import get_data_path_list, log_norm, mask_from_lens, maximum_path, recursive_munch


def _gb(value):
    return float(value) / 1024 ** 3


def _mps_line(label):
    if torch.backends.mps.is_available():
        print(
            '%s: cur=%.2f drv=%.2f max=%.2f'
            % (
                label,
                _gb(torch.mps.current_allocated_memory()),
                _gb(torch.mps.driver_allocated_memory()),
                _gb(torch.mps.recommended_max_memory()),
            ),
            flush=True,
        )


def _pad_last(x, width, value=0):
    if x.size(-1) == width:
        return x
    if x.size(-1) > width:
        return x[..., :width]
    return F.pad(x, (0, width - x.size(-1)), value=value)


def _time_gather(x, starts, length):
    '''Gather a fixed-width time crop with tensor starts, preserving gradients.'''
    base = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(0)
    idx = starts.to(x.device, dtype=torch.long).unsqueeze(1) + base
    while idx.ndim < x.ndim:
        idx = idx.unsqueeze(1)
    idx = idx.expand(*x.shape[:-1], length)
    return torch.gather(x, dim=-1, index=idx)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default='configs/config_vctk_ft.yml')
    parser.add_argument('--name', default='vctk_spksep')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--aligner-device', default='cpu')
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--log-interval', type=int, default=25)
    parser.add_argument('--fixed-pad', action='store_true', help='Pad texts/mels to one global size before the text path')
    parser.add_argument('--fixed-text-width', type=int, default=512)
    parser.add_argument('--fixed-mel-width', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--include-decoder-loss', action='store_true', help='Also run fixed-window F0Ntrain -> decoder -> STFT backward like joint Stage 2')
    parser.add_argument('--mel-len', type=int, default=64)
    parser.add_argument('--skip-duration-loss', action='store_true', help='Skip variable-length duration CE/L1 loop')
    parser.add_argument(
        '--crop-mode',
        choices=['random-slice', 'fixed-start', 'gather'],
        default='random-slice',
        help='Decoder-window crop strategy: random-slice (per-sample python-int slice, current); '
             'fixed-start (start=0 constant); gather (tensor-index gather, data-driven). '
             'Run each with --fixed-pad so crop-mode is the only variable.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config_path = (REPO_ROOT / args.config_path).resolve()
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ts._resolve_runtime_paths(config, str(config_path), name=args.name)

    device = args.device
    aligner_device = args.aligner_device
    train_list, val_list = get_data_path_list(config['data_params']['train_data'], config['data_params']['val_data'])
    loader = build_dataloader(
        train_list,
        config['data_params']['root_path'],
        OOD_data=config['data_params']['OOD_data'],
        min_length=config['data_params']['min_length'],
        batch_size=int(config.get('batch_size', 4)),
        num_workers=0,
        device=device,
        dataset_config={},
    )

    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    pitch_extractor = load_F0_models(config['F0_path'])
    plbert = load_plbert(config['PLBERT_dir'], vocab_size=config['model_params']['n_token'])
    model_params = recursive_munch(config['model_params'])
    loss_params = Munch(config['loss_params'])
    diffusion_enabled = loss_params.get('lambda_diff', 0) > 0 and loss_params.diff_epoch < config.get('epochs_2nd', 20)
    model = build_model(model_params, text_aligner, pitch_extractor, plbert, enable_diffusion=diffusion_enabled)
    for key in model:
        model[key].to(device)
    model.text_aligner.to(aligner_device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = ts.find_latest_second_stage_checkpoint(config['log_dir'])
        if latest is not None:
            checkpoint = latest[1]
    if checkpoint:
        print(f'Loading checkpoint: {checkpoint}', flush=True)
        model, _, _, _ = load_checkpoint(model, None, checkpoint, load_only_params=True)
        model.text_aligner.to(aligner_device)

    train_modules = [model.text_encoder, model.bert, model.bert_encoder, model.predictor, model.predictor_encoder]
    if args.include_decoder_loss:
        train_modules.extend([model.style_encoder, model.decoder])
    params = []
    for module in train_modules:
        module.train()
        params.extend(list(module.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    n_down = model.text_aligner.n_down
    stft_loss = MultiResolutionSTFTLoss().to(device) if args.include_decoder_loss else None

    print(
        'text-shape diagnostic: steps=%d fixed_pad=%s fixed_text=%d fixed_mel=%d include_decoder_loss=%s crop_mode=%s'
        % (args.steps, args.fixed_pad, args.fixed_text_width, args.fixed_mel_width, args.include_decoder_loss, args.crop_mode),
        flush=True,
    )
    _mps_line('start')

    seen = set()
    loader_iter = iter(loader)
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        waves = batch[0]
        tensors = [b.to(device) for b in batch[1:-2]]
        texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = tensors

        if args.fixed_pad:
            texts = _pad_last(texts, args.fixed_text_width)
            mels = _pad_last(mels, args.fixed_mel_width)
            mel_input_length = torch.minimum(mel_input_length, torch.full_like(mel_input_length, args.fixed_mel_width))
            input_lengths = torch.minimum(input_lengths, torch.full_like(input_lengths, args.fixed_text_width))

        shape_key = (int(texts.size(1)), int(mels.size(2) // (2 ** n_down)), int(mel_input_length.max().item() // (2 ** n_down)))
        is_new = shape_key not in seen
        seen.add(shape_key)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            aligner_mels = mels.detach().to(aligner_device)
            aligner_texts = texts.to(aligner_device)
            aligner_lengths = mel_input_length.to(aligner_device)
            aligner_width = aligner_mels.size(2) // (2 ** n_down)
            mask = ts._length_to_width_mask(aligner_lengths // (2 ** n_down), aligner_width).to(aligner_device)
            _, _, s2s_attn = model.text_aligner(aligner_mels, mask, aligner_texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2).to(device)
            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        text_mask = ts._length_to_text_mask(input_lengths, texts.size(1)).to(device)
        t_en = model.text_encoder(texts, input_lengths, text_mask)
        asr = t_en @ s2s_attn_mono
        d_gt = s2s_attn_mono.sum(axis=-1).detach()
        bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s_dur = model.predictor_encoder(ref_mels.detach().unsqueeze(1))
        d, p = model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

        # Exercise backward through the same variable-shaped text/attention graph,
        # plus a lightweight duration target loop similar to train_second.
        loss = asr.square().mean() * 0.01 + p.square().mean() * 0.01 + d.square().mean() * 0.001
        decoder_tensors = []
        if args.include_decoder_loss:
            mel_len = args.mel_len
            if int(mel_input_length.min().item() / 2) <= mel_len + 1:
                continue
            # Pick per-sample crop starts (frame resolution). fixed-start pins the
            # begin offset to a constant; the others vary it.
            starts = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                if args.crop_mode == 'fixed-start':
                    starts.append(0)
                else:
                    starts.append(int(np.random.randint(0, mel_length - mel_len)))

            # wav is numpy on CPU -> crop on CPU regardless of mode (no MPS op, no grad).
            wav = []
            for bib, s0 in enumerate(starts):
                y = waves[bib][(s0 * 2) * 300 : ((s0 + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).float())
            wav = torch.stack(wav).float().to(device).detach().unsqueeze(1)

            if args.crop_mode == 'gather':
                starts_t = torch.as_tensor(starts, device=device, dtype=torch.long)
                asr_source = _pad_last(asr, args.fixed_mel_width // 2)
                p_source = _pad_last(p, args.fixed_mel_width // 2)
                mel_source = _pad_last(mels, args.fixed_mel_width)
                en = _time_gather(asr_source, starts_t, mel_len)            # carries grad -> predictor/text
                p_en = _time_gather(p_source, starts_t, mel_len)            # carries grad -> predictor
                gt = _time_gather(mel_source, starts_t * 2, mel_len * 2).detach()
            else:
                # random-slice / fixed-start: per-sample python-int strided slice (begin is a graph constant)
                en = torch.stack([asr[bib, :, s0 : s0 + mel_len] for bib, s0 in enumerate(starts)])
                p_en = torch.stack([p[bib, :, s0 : s0 + mel_len] for bib, s0 in enumerate(starts)])
                gt = torch.stack(
                    [mels[bib, :, (s0 * 2) : ((s0 + mel_len) * 2)] for bib, s0 in enumerate(starts)]
                ).detach()
            with torch.no_grad():
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            s = model.style_encoder(ref_mels.detach().unsqueeze(1))
            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            y_rec = model.decoder(en, F0_fake, N_fake, s)
            loss = loss + 5.0 * stft_loss(y_rec, wav) + F.smooth_l1_loss(F0_real, F0_fake) / 10 + F.smooth_l1_loss(N_real, N_fake)
            decoder_tensors = [en, p_en, gt, wav, F0_real, N_real, s, F0_fake, N_fake, y_rec]
        loss_dur = d.new_tensor(0.0)
        loss_ce = d.new_tensor(0.0)
        if not args.skip_duration_loss:
            for _s2s_pred, _text_input, _text_length in zip(d, d_gt, input_lengths):
                _s2s_pred = _s2s_pred[:_text_length, :]
                _text_input = _text_input[:_text_length].long()
                _s2s_trg = torch.zeros_like(_s2s_pred)
                for pidx in range(_s2s_trg.shape[0]):
                    _s2s_trg[pidx, : _text_input[pidx]] = 1
                _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                loss_dur = loss_dur + F.l1_loss(_dur_pred[1 : _text_length - 1], _text_input[1 : _text_length - 1])
                loss_ce = loss_ce + F.binary_cross_entropy_with_logits(_s2s_pred.flatten(), _s2s_trg.flatten())
            loss = loss + 0.01 * (loss_dur + loss_ce) / texts.size(0)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        del batch, waves, tensors, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels
        for tensor in decoder_tensors:
            del tensor
        del s2s_attn, s2s_attn_mono, text_mask, t_en, asr, d_gt, bert_dur, d_en, s_dur, d, p, loss, loss_dur, loss_ce, decoder_tensors

        if step == 1 or step % args.log_interval == 0 or is_new:
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            marker = 'NEW' if is_new else 'old'
            _mps_line(f'step{step}_{marker}_shapes={len(seen)}_key={shape_key}_loss={last_loss:.6g}')

    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    _mps_line(f'end_unique_shapes={len(seen)}')


if __name__ == '__main__':
    main()
