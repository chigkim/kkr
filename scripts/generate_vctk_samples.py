#!/usr/bin/env python3
'''Generate Kokoro inference samples for multiple extracted VCTK voicepacks.'''

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'kokoro'))

from kokoro import KModel, KPipeline
from scripts.test_inference import convert_checkpoint

DEFAULT_SENTENCES = [
    'Please call Stella and ask her to bring these things with her from the store.',
    'The quick brown fox jumps over the lazy dog.',
    'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow.',
]


def resolve_device(device: str) -> str:
    if device == 'auto':
        return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    return device


def main():
    parser = argparse.ArgumentParser(description='Generate samples for VCTK Kokoro voicepacks')
    parser.add_argument('--checkpoint', default='StyleTTS2/logs/kokoro-vctk/epoch_2nd_00009.pth', help='StyleTTS2 Stage 2 checkpoint')
    parser.add_argument('--model', default=None, help='Already converted Kokoro weights; skips conversion if provided')
    parser.add_argument('--voices-dir', default='voices/vctk_samples', help='Directory containing speaker .pt voicepacks')
    parser.add_argument('--speakers', default='p239,p261,p303,p307', help='Comma-separated speaker IDs')
    parser.add_argument('--config', default='dataset/config.json', help='Kokoro config.json (prepare_dataset.py emits one with the expanded vocab)')
    parser.add_argument('--output-dir', default='generated/vctk_samples', help='Output directory')
    parser.add_argument('--lang-code', default='en-us', help='Language code. For --g2p-backend phonemizer, use a phonemizer/espeak language such as en-us, en-gb, de, fr-fr.')
    parser.add_argument('--g2p-backend', default='kokoro', choices=['kokoro', 'phonemizer'], help='Inference G2P backend; use phonemizer to match phonemizer/espeak training lists')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Inference device')
    args = parser.parse_args()

    out = REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.model:
        model_path = REPO_ROOT / args.model
    else:
        model_path = out / (Path(args.checkpoint).stem + '_kokoro.pth')
        if not model_path.exists():
            convert_checkpoint(str(REPO_ROOT / args.checkpoint), str(model_path))

    device = resolve_device(args.device)
    print(f'Using {device} device')
    print(f'Loading model: {model_path}')
    model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(REPO_ROOT / args.config), model=str(model_path)).to(device).eval()
    pipeline = KPipeline(lang_code=args.lang_code, repo_id='hexgrad/Kokoro-82M', model=model, g2p_backend=args.g2p_backend)

    for speaker in [s.strip() for s in args.speakers.split(',') if s.strip()]:
        voice_path = REPO_ROOT / args.voices_dir / f'{speaker}.pt'
        voice = torch.load(voice_path, map_location='cpu', weights_only=True)
        speaker_dir = out / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        print(f'\nSpeaker {speaker}: {voice_path}')

        for index, text in enumerate(DEFAULT_SENTENCES, 1):
            audio_chunks = []
            for _gs, phonemes, audio in pipeline(text, voice=voice, speed=1):
                print(f'  sample {index}: {phonemes[:80]}')
                audio_chunks.append(audio)

            if not audio_chunks:
                print(f'  WARNING: no audio for sample {index}')
                continue

            wav = np.concatenate(audio_chunks)
            wav_path = speaker_dir / f'sample_{index:02d}.wav'
            sf.write(str(wav_path), wav, 24000)
            print(f'  saved {wav_path.relative_to(REPO_ROOT)} ({len(wav) / 24000:.1f}s)')


if __name__ == '__main__':
    main()
