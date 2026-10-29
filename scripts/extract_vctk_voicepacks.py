#!/usr/bin/env python3
'''Extract Kokoro voicepacks for selected speakers from a training run.'''

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


def display_path(path: Path) -> str:
    '''Repo-relative path when possible, absolute otherwise.'''
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: str) -> Path:
    '''Absolute paths as-is; relative paths from the current directory, then the repo root.'''
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def latest_checkpoint(run_dir: Path, pattern: str) -> Path | None:
    checkpoints = []
    for path in glob.glob(str(run_dir / pattern)):
        match = re.search(r'_(\d+)\.pth$', Path(path).name)
        if match:
            checkpoints.append((int(match.group(1)), Path(path)))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


def resolve_checkpoint(run_dir: Path, path: str | None, names: list[str], latest_pattern: str) -> Path:
    '''Explicit path, else the first of `names` in run_dir, else the latest epoch checkpoint.'''
    if path:
        candidate = resolve_path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f'Checkpoint not found: {candidate}')

    for name in names:
        candidate = run_dir / name
        if candidate.exists():
            return candidate

    latest = latest_checkpoint(run_dir, latest_pattern)
    if latest is not None:
        print(f'Using latest checkpoint: {display_path(latest)}')
        return latest

    raise FileNotFoundError(f'No checkpoint matching {names} or {latest_pattern} in {run_dir}')


def main():
    parser = argparse.ArgumentParser(description='Extract voicepacks for selected speakers')
    parser.add_argument('--run-dir', default='runs/vctk', help='Training run directory with best_1st.pth / best_2nd.pth')
    parser.add_argument('--stage', choices=['stage1', 'stage2'], default='stage2', help='Use the best Stage 1 or best Stage 2 checkpoint')
    parser.add_argument('--model', default=None, help='Override checkpoint used for predictor_encoder/model')
    parser.add_argument('--style-encoder-model', default=None, help='Override checkpoint used for style_encoder (default: same as --model; pass a Stage 1 checkpoint for the hybrid)')
    parser.add_argument('--dataset-root', default='../voices/vckt-24k/24k', help='Root containing speaker WAV folders')
    parser.add_argument('--speakers', default='p239,p261,p303,p307', help='Comma-separated speaker IDs')
    parser.add_argument('--output-dir', default=None, help='Output directory for .pt voicepacks (default: voices/<run>_<stage>)')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of WAVs per speaker to average')
    parser.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'cuda'], help='Extraction device')
    parser.add_argument('--style-mode', default='medoid', choices=['medoid', 'mean', 'mean-renorm'], help='How to combine per-utterance styles')
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    if args.stage == 'stage1':
        model_path = resolve_checkpoint(run_dir, args.model, ['best_1st.pth', 'first_stage.pth'], 'epoch_1st_*.pth')
    else:
        model_path = resolve_checkpoint(run_dir, args.model, ['best_2nd.pth'], 'epoch_2nd_*.pth')
    # The decoder trains together with style_encoder in Stage 2, so pair them
    # by default; a Stage 1 encoder is only used when asked for explicitly.
    style_encoder_path = resolve_path(args.style_encoder_model) if args.style_encoder_model else model_path
    if not style_encoder_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {style_encoder_path}')
    output_dir = resolve_path(args.output_dir) if args.output_dir else REPO_ROOT / 'voices' / f'{run_dir.name}_{args.stage}'

    print(f'Using model checkpoint: {display_path(model_path)}')
    print(f'Using style encoder:    {display_path(style_encoder_path)}')

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = resolve_path(args.dataset_root)

    for speaker in [s.strip() for s in args.speakers.split(',') if s.strip()]:
        audio_dir = dataset_root / speaker
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

    print(f'\nDone. Voicepacks saved to: {display_path(output_dir)}')


if __name__ == '__main__':
    main()
