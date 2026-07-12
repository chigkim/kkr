#!/usr/bin/env python3
'''Extract Kokoro voicepacks for selected VCTK speakers.'''

import argparse
import glob
import os
import re
from pathlib import Path
import sys

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_voicepack import extract_voicepack
LOG_DIR = REPO_ROOT / 'StyleTTS2/logs/kokoro-vctk'


def latest_checkpoint(pattern: str) -> Path | None:
    checkpoints = []
    for path in glob.glob(str(LOG_DIR / pattern)):
        match = re.search(r'_(\d+)\.pth$', Path(path).name)
        if match:
            checkpoints.append((int(match.group(1)), Path(path)))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_checkpoint(path: str | None, fallback: str | None = None) -> Path:
    if path:
        candidate = REPO_ROOT / path
        if candidate.exists():
            return candidate
        print(f'WARNING: checkpoint not found: {candidate}')

    if fallback:
        candidate = REPO_ROOT / fallback
        if candidate.exists():
            print(f'Using fallback checkpoint: {candidate.relative_to(REPO_ROOT)}')
            return candidate

    latest = latest_checkpoint('epoch_2nd_*.pth')
    if latest is not None:
        print(f'Using latest Stage 2 checkpoint: {latest.relative_to(REPO_ROOT)}')
        return latest

    raise FileNotFoundError(f'No checkpoint found for {path!r}')


def main():
    parser = argparse.ArgumentParser(description='Extract voicepacks for VCTK speakers')
    parser.add_argument('--stage', choices=['stage1', 'stage2'], default='stage2', help='Use best Stage 1 only, or best Stage 2 + best Stage 1 style encoder')
    parser.add_argument('--model', default=None, help='Override checkpoint used for predictor_encoder/model')
    parser.add_argument('--style-encoder-model', default=None, help='Override checkpoint used for style_encoder')
    parser.add_argument('--dataset-root', default='../voices/vckt-24k/24k', help='Root containing speaker WAV folders')
    parser.add_argument('--speakers', default='p239,p261,p303,p307', help='Comma-separated speaker IDs')
    parser.add_argument('--output-dir', default=None, help='Output directory for .pt voicepacks')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of WAVs per speaker to average')
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'cuda'], help='Extraction device')
    parser.add_argument('--style-mode', default='medoid', choices=['medoid', 'mean', 'mean-renorm'], help='How to combine per-utterance styles (plain mean dilutes speaker identity)')
    args = parser.parse_args()

    if args.stage == 'stage1':
        model_path = resolve_checkpoint(args.model or 'StyleTTS2/logs/kokoro-vctk/best_1st.pth', 'StyleTTS2/logs/kokoro-vctk/first_stage.pth')
        style_encoder_path = resolve_checkpoint(args.style_encoder_model or str(model_path.relative_to(REPO_ROOT)), 'StyleTTS2/logs/kokoro-vctk/first_stage.pth')
        output_dir = REPO_ROOT / (args.output_dir or 'voices/vctk_best_stage1')
    else:
        model_path = resolve_checkpoint(args.model or 'StyleTTS2/logs/kokoro-vctk/best_2nd.pth')
        style_encoder_path = resolve_checkpoint(args.style_encoder_model or 'StyleTTS2/logs/kokoro-vctk/best_1st.pth', 'StyleTTS2/logs/kokoro-vctk/first_stage.pth')
        output_dir = REPO_ROOT / (args.output_dir or 'voices/vctk_samples')

    print(f'Using model checkpoint: {model_path.relative_to(REPO_ROOT)}')
    print(f'Using style encoder:    {style_encoder_path.relative_to(REPO_ROOT)}')

    output_dir.mkdir(parents=True, exist_ok=True)

    for speaker in [s.strip() for s in args.speakers.split(',') if s.strip()]:
        audio_dir = REPO_ROOT / args.dataset_root / speaker
        output_path = output_dir / f'{speaker}.pt'
        print(f'\n=== Extracting {speaker} ===')
        extract_voicepack(
            model_path=str(model_path),
            audio_dir=str(audio_dir),
            output_path=str(output_path),
            num_samples=args.num_samples,
            device=args.device,
            style_encoder_model=str(style_encoder_path),
            style_mode=args.style_mode,
        )

    print(f'\nDone. Voicepacks saved to: {output_dir.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
