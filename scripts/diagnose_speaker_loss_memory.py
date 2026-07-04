#!/usr/bin/env python3
'''Exercise only Stage 2 prosody speaker-separation modules and report MPS memory.'''

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
import yaml
from munch import Munch

import train_second as ts
from meldataset import build_dataloader
from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
from losses import DiscriminatorLoss, GeneratorLoss, WavLMLoss
from speaker_losses import SpeakerCenters, SpeakerClassifier, speaker_separation_loss
from Utils.PLBERT.util import load_plbert
from utils import get_data_path_list, recursive_munch


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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', default='configs/config_vctk_ft.yml')
    parser.add_argument('--name', default='vctk_spksep')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint to load; default latest epoch_2nd_*.pth')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--no-optimizer', action='store_true', help='Run backward but skip optimizer.step()')
    parser.add_argument('--centers-only', action='store_true', help='Detach predictor_encoder output; train only classifier/centers')
    parser.add_argument('--random-input', action='store_true', help='Use random fixed-shape ref_mels instead of dataloader ref_mels')
    parser.add_argument('--random-labels', action='store_true', help='Use random speaker labels to keep CE/center losses non-trivial')
    parser.add_argument('--lambda-scale', type=float, default=1.0, help='Scale configured speaker loss weights for stress testing')
    parser.add_argument('--include-adversarial', action='store_true', help='Also run MPD/MSD discriminator and generator losses on MPS')
    parser.add_argument('--include-wavlm', action='store_true', help='Also run frozen WavLM perceptual loss on MPS')
    parser.add_argument('--audio-segment-size', type=int, default=16000, help='Fixed audio length for adversarial/WavLM diagnostics')
    return parser.parse_args()


def main():
    args = _parse_args()
    config_path = (REPO_ROOT / args.config_path).resolve()
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    ts._resolve_runtime_paths(config, str(config_path), name=args.name)

    device = args.device
    train_list, val_list = get_data_path_list(config['data_params']['train_data'], config['data_params']['val_data'])
    train_loader = build_dataloader(
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

    num_speakers = int(config.get('num_speakers') or ts._num_speakers_from_lists(train_list, val_list))
    style_dim = int(config['model_params'].get('style_dim', 128))
    classifier = SpeakerClassifier(style_dim, num_speakers).to(device)
    centers = SpeakerCenters(num_speakers, style_dim).to(device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        latest = ts.find_latest_second_stage_checkpoint(config['log_dir'])
        if latest is not None:
            checkpoint = latest[1]
    if checkpoint:
        print(f'Loading checkpoint: {checkpoint}', flush=True)
        # Loading the whole model is useful because predictor_encoder weights affect
        # its graph/param layout. Classifier/center state is loaded below if present.
        model['prosody_speaker_classifier'] = classifier
        model['prosody_speaker_centers'] = centers
        model, _, _, _ = load_checkpoint(model, None, checkpoint, load_only_params=True)
        classifier = model['prosody_speaker_classifier'].to(device)
        centers = model['prosody_speaker_centers'].to(device)

    modules = [classifier, centers]
    if not args.centers_only:
        modules.append(model.predictor_encoder)
    audio_param = None
    gl = None
    dl = None
    wl = None
    disc_optimizer = None
    if args.include_adversarial:
        model.mpd.to(device)
        model.msd.to(device)
        gl = GeneratorLoss(model.mpd, model.msd).to(device)
        dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    if args.include_wavlm:
        model.wd.to(device)
        wl = WavLMLoss(
            model_params.slm.model,
            model.wd,
            config['preprocess_params'].get('sr', 24000),
            model_params.slm.sr,
        ).to(device)
    if args.include_adversarial or args.include_wavlm:
        audio_param = torch.nn.Parameter(
            torch.randn(int(config.get('batch_size', 4)), 1, args.audio_segment_size, device=device) * 0.01
        )

    params = []
    for module in modules:
        params.extend(list(module.parameters()))
    if audio_param is not None:
        params.append(audio_param)
    optimizer = None if args.no_optimizer else torch.optim.AdamW(params, lr=args.lr)
    if args.include_adversarial and not args.no_optimizer:
        disc_optimizer = torch.optim.AdamW(list(model.mpd.parameters()) + list(model.msd.parameters()), lr=args.lr)

    model.predictor_encoder.train()
    classifier.train()
    centers.train()
    model.mpd.train()
    model.msd.train()
    if wl is not None:
        wl.train()

    print(
        'speaker diagnostic: steps=%d no_optimizer=%s centers_only=%s random_input=%s random_labels=%s include_adversarial=%s include_wavlm=%s audio_segment_size=%d num_speakers=%d'
        % (
            args.steps,
            args.no_optimizer,
            args.centers_only,
            args.random_input,
            args.random_labels,
            args.include_adversarial,
            args.include_wavlm,
            args.audio_segment_size,
            num_speakers,
        ),
        flush=True,
    )
    _mps_line('start')

    loader_iter = iter(train_loader)
    last_loss = 0.0
    for step in range(1, args.steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        labels = batch[-2].to(device)
        ref_labels = batch[-1].to(device)
        if args.random_input:
            # Same fixed shape as Collater ref_mels: [B, 80, 192].
            ref_mels = torch.randn(labels.size(0), 80, 192, device=device)
        else:
            ref_mels = batch[7].to(device)  # batch[1:-2][6] == ref_mels

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        style = model.predictor_encoder(ref_mels.detach().unsqueeze(1))
        if args.centers_only:
            style = style.detach()
        speaker_ids = ref_labels if model_params.multispeaker else labels
        if args.random_labels:
            speaker_ids = torch.randint(0, num_speakers, speaker_ids.shape, device=device)
        loss, ce, center, margin = speaker_separation_loss(
            style,
            speaker_ids,
            classifier=classifier,
            centers=centers,
            lambda_ce=loss_params.get('lambda_prosody_spk', 0) * args.lambda_scale,
            lambda_center=loss_params.get('lambda_prosody_center', 0) * args.lambda_scale,
            lambda_margin=loss_params.get('lambda_speaker_margin', 0) * args.lambda_scale,
            margin=loss_params.get('speaker_center_margin', 1.0),
        )

        wav = None
        y_rec = None
        d_loss = None
        loss_gen = None
        loss_lm = None
        if audio_param is not None:
            batch_size = labels.size(0)
            y_rec = audio_param[:batch_size]
            # Fixed-shape synthetic target. This isolates adversarial/WavLM memory
            # behavior from decoder/text/aligner variability while preserving
            # gradients through generated audio.
            wav = torch.randn_like(y_rec).detach()

        if args.include_adversarial:
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
            d_loss = dl(wav.detach(), y_rec.detach()).mean()
            d_loss.backward()
            if disc_optimizer is not None:
                disc_optimizer.step()
            # Generator adversarial loss should backprop through y_rec/audio_param,
            # not update discriminator params in this diagnostic step.
            for p in model.mpd.parameters():
                p.requires_grad_(False)
            for p in model.msd.parameters():
                p.requires_grad_(False)
            loss_gen = gl(wav.detach(), y_rec).mean()
            for p in model.mpd.parameters():
                p.requires_grad_(True)
            for p in model.msd.parameters():
                p.requires_grad_(True)
            loss = loss + loss_gen

        if args.include_wavlm:
            loss_lm = wl(wav.squeeze(1), y_rec.squeeze(1)).mean()
            loss = loss + loss_params.get('lambda_slm', 1.0) * loss_lm

        loss.backward()
        if optimizer is not None:
            optimizer.step()
        last_loss = float(loss.detach().cpu())
        adv_value = 0.0 if loss_gen is None else float(loss_gen.detach().cpu())
        wavlm_value = 0.0 if loss_lm is None else float(loss_lm.detach().cpu())

        del batch, labels, ref_labels, ref_mels, style, speaker_ids, loss, ce, center, margin
        if wav is not None:
            del wav, y_rec
        if d_loss is not None:
            del d_loss
        if loss_gen is not None:
            del loss_gen
        if loss_lm is not None:
            del loss_lm
        if step % args.log_interval == 0 or step == 1:
            gc.collect()
            if device == 'mps':
                torch.mps.empty_cache()
            _mps_line(f'step{step}_loss={last_loss:.6g}_adv={adv_value:.6g}_wavlm={wavlm_value:.6g}')

    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    _mps_line('end')


if __name__ == '__main__':
    main()
