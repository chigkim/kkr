#!/usr/bin/env python3
'''Repeat Stage 2 validation to diagnose MPS driver/native memory growth.'''

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

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from munch import Munch

import train_second as ts
from losses import MultiResolutionSTFTLoss
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
from speaker_losses import SpeakerCenters, SpeakerClassifier
from symbols_config import apply_symbols_config
from Utils.PLBERT.util import load_plbert
from utils import (
    get_data_path_list,
    mask_from_lens,
    maximum_path,
    recursive_munch,
)


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
    else:
        print(f'{label}: MPS unavailable', flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default='configs/config_vctk_ft.yml')
    parser.add_argument('--name', default='vctk_spksep')
    parser.add_argument('--cycles', type=int, default=5, help='Validation cycles to run')
    parser.add_argument('--batches', type=int, default=12, help='Validation batches per cycle')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--eval-device', default=None, help='Defaults to config eval_device or CPU on MPS')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to load; default latest epoch_2nd_*.pth')
    parser.add_argument(
        '--include-raw-style-loop',
        action='store_true',
        help='Run the old validation raw-length predictor/style loop to reproduce suspected MPSGraph growth.',
    )
    parser.add_argument(
        '--no-roundtrip',
        action='store_true',
        help='Do not move model back to training device between cycles.',
    )
    return parser.parse_args()


def _build_state(args):
    config_path = (REPO_ROOT / args.config_path).resolve()
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ts._resolve_runtime_paths(config, str(config_path), name=args.name)
    apply_symbols_config(config, str(config_path))

    device = args.device
    eval_device = args.eval_device or config.get('eval_device', 'cpu' if device == 'mps' else device)
    aligner_device = config.get('aligner_device', 'cpu' if device == 'mps' else device)
    discriminator_device = config.get('discriminator_device', device)
    batch_size = int(config.get('batch_size', 4))
    n_down = int(config.get('n_down', 1))

    train_list, val_list = get_data_path_list(
        config['data_params']['train_data'],
        config['data_params']['val_data'],
    )
    val_loader = build_dataloader(
        val_list,
        config['data_params']['root_path'],
        OOD_data=config['data_params']['OOD_data'],
        min_length=config['data_params']['min_length'],
        batch_size=batch_size,
        validation=True,
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

    num_speakers = int(config.get('num_speakers') or ts._num_speakers_from_lists(train_list, val_list))
    style_dim = int(config['model_params'].get('style_dim', 128))
    if loss_params.get('lambda_prosody_spk', 0) > 0:
        model['prosody_speaker_classifier'] = SpeakerClassifier(style_dim, num_speakers).to(device)
    if loss_params.get('lambda_prosody_center', 0) > 0 or loss_params.get('lambda_speaker_margin', 0) > 0:
        model['prosody_speaker_centers'] = SpeakerCenters(num_speakers, style_dim).to(device)

    if aligner_device != device:
        model.text_aligner.to(aligner_device)
    if discriminator_device != device:
        model.mpd.to(discriminator_device)
        model.msd.to(discriminator_device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = ts.find_latest_second_stage_checkpoint(config['log_dir'])
        if latest is None:
            raise FileNotFoundError(f'No epoch_2nd_*.pth checkpoints found in {config["log_dir"]}')
        checkpoint = latest[1]
    print(f'Loading checkpoint: {checkpoint}', flush=True)
    model, _, _, _ = load_checkpoint(model, None, checkpoint, load_only_params=True)

    if aligner_device != device:
        model.text_aligner.to(aligner_device)
    if discriminator_device != device:
        model.mpd.to(discriminator_device)
        model.msd.to(discriminator_device)

    stft_loss = MultiResolutionSTFTLoss().to(device)
    return config, model_params, model, stft_loss, val_loader, device, eval_device, aligner_device, discriminator_device, n_down


def _run_validation_cycle(
    config,
    model_params,
    model,
    stft_loss,
    val_loader,
    eval_device,
    aligner_device,
    n_down,
    max_batches,
    include_raw_style_loop,
):
    total_loss = 0.0
    iters = 0
    fixed_mel_len = int(config.get('fixed_training_mel_len_frames', 64))

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
            waves = batch[0]
            batch = [b.to(eval_device) for b in batch[1:-2]]
            texts, input_lengths, _ref_texts, _ref_lengths, mels, mel_input_length, ref_mels = batch

            aligner_mels = mels.detach().to(aligner_device)
            aligner_width = aligner_mels.size(2) // (2**n_down)
            mask = ts._length_to_width_mask(
                mel_input_length.to(aligner_device) // (2**n_down),
                aligner_width,
            ).to(aligner_device)
            text_mask = ts._length_to_text_mask(input_lengths, texts.size(1)).to(texts.device)

            _, _, s2s_attn = model.text_aligner(aligner_mels, mask, texts.to(aligner_device))
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2).to(eval_device)
            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2**n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = model.text_encoder(texts, input_lengths, text_mask)
            asr = t_en @ s2s_attn_mono

            if include_raw_style_loop:
                # Reproduce the old validation path: raw unbucketed mel lengths.
                # This is intentionally suspect for MPS driver growth.
                ss = []
                for bib in range(len(mel_input_length)):
                    mel = mels[bib, :, : mel_input_length[bib]]
                    ss.append(model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1)))
                s_dur_for_predictor = torch.stack(ss).squeeze(1)
            else:
                # Stable validation path: same fixed-shape reference mel path as training.
                s_dur_for_predictor = model.predictor_encoder(
                    ref_mels.detach().unsqueeze(1) if model_params.multispeaker else mels.detach().unsqueeze(1)
                )

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            _d, p = model.predictor(d_en, s_dur_for_predictor, input_lengths, s2s_attn_mono, text_mask)

            mel_len = min(int(mel_input_length.min().item() / 2 - 1), fixed_mel_len)
            if mel_len < 16:
                continue

            en = []
            gt = []
            p_en = []
            wav = []
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(asr[bib, :, random_start : random_start + mel_len])
                p_en.append(p[bib, :, random_start : random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2) : ((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).float().to(eval_device))

            wav = torch.stack(wav).float().detach()
            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()

            s_dur = model.predictor_encoder(
                ref_mels.detach().unsqueeze(1) if model_params.multispeaker else gt.unsqueeze(1)
            )
            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)
            s = model.style_encoder(
                ref_mels.detach().unsqueeze(1) if model_params.multispeaker else gt.unsqueeze(1)
            )
            y_rec = model.decoder(en, F0_fake, N_fake, s)
            total_loss += float(stft_loss(y_rec.squeeze(1), wav.detach()).mean())
            iters += 1

    return total_loss / max(iters, 1), iters


def main():
    args = _parse_args()
    config, model_params, model, stft_loss, val_loader, device, eval_device, aligner_device, discriminator_device, n_down = _build_state(args)

    _mps_line('after_load')
    for cycle in range(1, args.cycles + 1):
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
        _mps_line(f'cycle{cycle}_start')

        if eval_device != device:
            for key in model:
                model[key].to(eval_device)
            stft_loss = stft_loss.to(eval_device)
        for key in model:
            model[key].eval()

        loss, iters = _run_validation_cycle(
            config,
            model_params,
            model,
            stft_loss,
            val_loader,
            eval_device,
            aligner_device,
            n_down,
            args.batches,
            args.include_raw_style_loop,
        )
        print(f'cycle{cycle}: val_batches={iters} loss={loss:.4f}', flush=True)
        _mps_line(f'cycle{cycle}_after_val')

        if eval_device != device and not args.no_roundtrip:
            for key in model:
                model[key].to(device)
            if aligner_device != device:
                model.text_aligner.to(aligner_device)
            if discriminator_device != device:
                model.mpd.to(discriminator_device)
                model.msd.to(discriminator_device)
            stft_loss = stft_loss.to(device)
        for key in model:
            model[key].train()

        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
        _mps_line(f'cycle{cycle}_after_back_to_train_device')


if __name__ == '__main__':
    main()
