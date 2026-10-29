#!/usr/bin/env python3
'''Measure whether extracted voicepacks reproduce their speakers.

Synthesizes a few sentences per voicepack with a converted checkpoint, embeds
the audio with a WavLM speaker-verification model, and compares it against
real recordings of the same speakers:

- top-1 speaker ID: how often generated audio is closest to its own speaker's
  real-recording centroid (real held-out recordings give the ceiling)
- between-speaker cosine: how alike different generated voices sound (lower
  is more distinct; compare with the real-recording value)

Usage:
    uv run scripts/speaker_similarity.py --checkpoint runs/af_v2/best_2nd.pth \
        --voices voices/af_v2_stage2 --audio-root af/en-us --out generated/af_v2_sv
'''

import argparse
import glob
import itertools
import json
import random
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'kokoro'))
sys.path.insert(0, str(REPO_ROOT))

from kokoro import KModel, KPipeline  # noqa: E402
from scripts.test_inference import convert_checkpoint  # noqa: E402
from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: E402

TEXTS = [
    'The birch canoe slid on the smooth planks.',
    'Glue the sheet to the dark blue background.',
    'These days a chicken leg is a rare dish.',
]
SV_MODEL = 'microsoft/wavlm-base-plus-sv'


def main():
    parser = argparse.ArgumentParser(description='Speaker-verification check for extracted voicepacks')
    parser.add_argument('--checkpoint', required=True, help='StyleTTS2 training checkpoint (e.g. best_2nd.pth)')
    parser.add_argument('--voices', required=True, help='Directory of <speaker>.pt voicepacks')
    parser.add_argument('--audio-root', default='af/en-us', help='Directory with one folder of real WAVs per speaker')
    parser.add_argument('--config', default='dataset/config.json', help='KModel config.json matching the checkpoint vocab')
    parser.add_argument('--lang', default='en-us', help='KPipeline language code for the test sentences (espeak code with --g2p-backend phonemizer)')
    parser.add_argument('--g2p-backend', default='phonemizer', choices=['phonemizer', 'kokoro'], help='Inference G2P; phonemizer matches lists from prepare_dataset.py')
    parser.add_argument('--out', required=True, help='Output directory for samples, converted model and results.json')
    parser.add_argument('--real-per-speaker', type=int, default=11, help='Real WAVs per speaker: 3 held out, the rest form the centroid')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
    sv = WavLMForXVector.from_pretrained(SV_MODEL).eval()

    @torch.no_grad()
    def embed(wav, sr):
        wav = librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=16000)
        x = fe(wav, sampling_rate=16000, return_tensors='pt')
        return F.normalize(sv(**x).embeddings[0], dim=0)

    def between(embs):
        keys = list(embs)
        return float(np.mean([
            float(a @ b)
            for i, j in itertools.combinations(range(len(keys)), 2)
            for a in embs[keys[i]] for b in embs[keys[j]]
        ]))

    voice_dir = Path(args.voices)
    speakers = sorted(Path(p).stem for p in glob.glob(str(voice_dir / '*.pt')))
    if len(speakers) < 2:
        raise SystemExit(f'Need at least two voicepacks in {voice_dir}')

    audio_root = Path(args.audio_root)
    real, centroid = {}, {}
    for s in speakers:
        files = sorted(glob.glob(str(audio_root / s / '*.wav')))
        if len(files) < 4:
            raise SystemExit(f'Need at least 4 real WAVs for {s} in {audio_root / s}')
        e = [embed(*sf.read(f)) for f in random.sample(files, min(args.real_per_speaker, len(files)))]
        real[s] = e[:3]
        centroid[s] = F.normalize(torch.stack(e[3:]).mean(0), dim=0)

    kpath = out / 'kmodel.pth'
    convert_checkpoint(str(args.checkpoint), str(kpath))
    model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(args.config), model=str(kpath)).eval()
    pipe = KPipeline(lang_code=args.lang, repo_id='hexgrad/Kokoro-82M', model=model, g2p_backend=args.g2p_backend)

    generated = {}
    for s in speakers:
        voice = torch.load(voice_dir / f'{s}.pt', weights_only=True)
        embs = []
        for k, text in enumerate(TEXTS):
            audio = np.concatenate([r.audio.numpy() for r in pipe(text, voice=voice)])
            if k == 0:
                sf.write(out / f'{s}.wav', audio, 24000)
            embs.append(embed(audio, 24000))
        generated[s] = embs

    C = torch.stack([centroid[s] for s in speakers])
    correct, ranks, real_correct = 0, {}, 0
    for i, s in enumerate(speakers):
        sims = C @ F.normalize(torch.stack(generated[s]).mean(0), dim=0)
        correct += int(sims.argmax() == i)
        ranks[s] = int((sims > sims[i]).sum()) + 1
        real_correct += int((C @ F.normalize(torch.stack(real[s]).mean(0), dim=0)).argmax() == i)

    n = len(speakers)
    results = {
        'checkpoint': str(args.checkpoint),
        'voices': str(voice_dir),
        'top1': correct,
        'top1_real': real_correct,
        'median_rank': float(np.median(list(ranks.values()))),
        'between_generated': between(generated),
        'between_real': between(real),
        'ranks': ranks,
    }
    print(f'\nSpeaker ID vs real centroids: top-1 {correct}/{n} (real recordings: {real_correct}/{n}), median rank {results["median_rank"]:g}/{n}')
    print(f'Between-speaker cosine: generated {results["between_generated"]:.3f}, real {results["between_real"]:.3f} (lower = more distinct)')
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(f'Results: {out / "results.json"}')


if __name__ == '__main__':
    main()
