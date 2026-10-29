#!/usr/bin/env python3
'''Measure whether extracted voicepacks reproduce their speakers.

Synthesizes a few sentences per voicepack with a converted checkpoint, embeds
the audio with a WavLM speaker-verification model, and compares it against
real recordings of the same speakers:

- top-1 speaker ID: how often generated audio is closest to its own speaker's
  real-recording centroid (real held-out recordings give the ceiling)
- between-speaker cosine: how alike different generated voices sound (lower
  is more distinct; compare with the real-recording value)
- margin: cosine to the speaker's own real centroid minus the highest cosine
  to another speaker's centroid (positive = identified; the mean is the
  primary summary, because it moves before top-1 does)
- centered between-speaker cosine: the same comparison after subtracting the
  mean enrollment embedding of the real recordings. This removes what real
  speakers share; a component shared only by generated voices (for example
  a synthetic-speech signature) remains and shows up as a positive value
- generated-vs-generated matrix: whether each generated voice is closer to
  itself on other texts than to any other generated voice (only pairs of
  different texts are compared, so sentence content cannot favour impostors)
- same-text control: between-speaker cosine of the real `*_001` recordings,
  which all read "Please call Stella.", and of generated audio of the same
  sentence for the same speakers. Descriptive only; the real files may also
  be in the enrollment set

Pass --real-files-from with an earlier results.json to reuse its enrollment
and held-out files, so two checkpoints are scored against the same recordings.

Usage:
    uv run scripts/speaker_similarity.py --checkpoint runs/af_v2/best_2nd.pth \
        --voices voices/af_v2_stage2 --audio-root af/en-us --out generated/af_v2_sv
'''

import argparse
import glob
import hashlib
import itertools
import json
import random
from pathlib import Path
import subprocess
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
SAME_TEXT = 'Please call Stella.'
SV_MODEL = 'microsoft/wavlm-base-plus-sv'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sv_revision():
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(SV_MODEL, local_files_only=True)).name
    except Exception:  # noqa: BLE001 - provenance only, never fatal
        return None


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
    parser.add_argument('--real-files-from', default=None, help='results.json whose real_files (enrollment and held-out recordings) to reuse')
    args = parser.parse_args()
    if args.real_per_speaker < 4:
        raise SystemExit('--real-per-speaker must be at least 4 (3 held out, at least 1 for the centroid)')
    reuse = json.loads(Path(args.real_files_from).read_text())['real_files'] if args.real_files_from else None

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

    def between(embs, offset=None):
        keys = list(embs)

        def prep(e):
            return e if offset is None else F.normalize(e - offset, dim=0)

        return float(np.mean([
            float(prep(a) @ prep(b))
            for i, j in itertools.combinations(range(len(keys)), 2)
            for a in embs[keys[i]] for b in embs[keys[j]]
        ]))

    voice_dir = Path(args.voices)
    speakers = sorted(Path(p).stem for p in glob.glob(str(voice_dir / '*.pt')))
    if len(speakers) < 2:
        raise SystemExit(f'Need at least two voicepacks in {voice_dir}')

    audio_root = Path(args.audio_root)
    real, centroid, real_files, real_all = {}, {}, {}, []
    for s in speakers:
        files = sorted(glob.glob(str(audio_root / s / '*.wav')))
        # Keep the recordings the voicepack was extracted from out of the
        # real-recording centroid and held-out set.
        used_path = voice_dir / f'{s}.files.json'
        if used_path.exists():
            used = {str(Path(f).resolve()) for f in json.loads(used_path.read_text(encoding='utf-8'))}
            files = [f for f in files if str(Path(f).resolve()) not in used]
        else:
            raise SystemExit(f'No {used_path}; extract with scripts/extract_voicepack.py so the extraction recordings can be excluded')
        if len(files) < 4:
            raise SystemExit(f'Need at least 4 real WAVs for {s} in {audio_root / s}')
        if reuse is not None:
            if s not in reuse:
                raise SystemExit(f'{s} is not in {args.real_files_from}')
            overlap = {str(Path(f).resolve()) for f in reuse[s]} & used
            if overlap:
                raise SystemExit(f'{len(overlap)} reused real files of {s} are in this extraction set')
            real_files[s] = reuse[s]
        else:
            real_files[s] = random.sample(files, min(args.real_per_speaker, len(files)))
        e = [embed(*sf.read(f)) for f in real_files[s]]
        real[s] = e[:3]
        centroid[s] = F.normalize(torch.stack(e[3:]).mean(0), dim=0)
        real_all.extend(e[3:])
    # Mean of the real enrollment embeddings, used for the centered cosine.
    offset = torch.stack(real_all).mean(0)

    same_text = {}
    for s in speakers:
        f = sorted(glob.glob(str(audio_root / s / f'{s}_001_*.wav')))
        if f:
            same_text[s] = [embed(*sf.read(f[0]))]

    kpath = out / 'kmodel.pth'
    convert_checkpoint(str(args.checkpoint), str(kpath))
    model = KModel(repo_id='hexgrad/Kokoro-82M', config=str(args.config), model=str(kpath)).eval()
    pipe = KPipeline(lang_code=args.lang, repo_id='hexgrad/Kokoro-82M', model=model, g2p_backend=args.g2p_backend)

    generated = {}
    for s in speakers:
        voice = torch.load(voice_dir / f'{s}.pt', weights_only=True)
        embs = []
        for k, text in enumerate(TEXTS):
            # Kokoro's source excitation draws torch noise; seed it so runs
            # repeat and every speaker gets the same noise for a given text.
            torch.manual_seed(args.seed + k)
            audio = np.concatenate([r.audio.numpy() for r in pipe(text, voice=voice)])
            if k == 0:
                sf.write(out / f'{s}.wav', audio, 24000)
            embs.append(embed(audio, 24000))
        generated[s] = embs

    generated_same_text = {}
    for s in same_text:
        voice = torch.load(voice_dir / f'{s}.pt', weights_only=True)
        torch.manual_seed(args.seed + len(TEXTS))
        audio = np.concatenate([r.audio.numpy() for r in pipe(SAME_TEXT, voice=voice)])
        generated_same_text[s] = [embed(audio, 24000)]

    def margin(sims, i):
        return float(sims[i] - torch.cat([sims[:i], sims[i + 1:]]).max())

    C = torch.stack([centroid[s] for s in speakers])
    correct, ranks, real_correct, margins, real_margins = 0, {}, 0, {}, {}
    for i, s in enumerate(speakers):
        sims = C @ F.normalize(torch.stack(generated[s]).mean(0), dim=0)
        correct += int(sims.argmax() == i)
        ranks[s] = int((sims > sims[i]).sum()) + 1
        margins[s] = margin(sims, i)
        real_sims = C @ F.normalize(torch.stack(real[s]).mean(0), dim=0)
        real_correct += int(real_sims.argmax() == i)
        real_margins[s] = margin(real_sims, i)

    # Generated vs generated, over pairs of different texts only: the same
    # sentence would make two voices look alike for reasons of content.
    n = len(speakers)
    G = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        a, b = generated[speakers[i]], generated[speakers[j]]
        G[i, j] = np.mean([float(x @ y) for p, x in enumerate(a) for q, y in enumerate(b) if p != q])
    gen_self = int(sum(G[i, i] > np.delete(G[i], i).max() for i in range(n)))
    gen_nearest = {s: speakers[int(np.argmax(np.where(np.arange(n) == i, -np.inf, G[i])))] for i, s in enumerate(speakers)}

    results = {
        'checkpoint': str(args.checkpoint),
        'voices': str(voice_dir),
        'seed': args.seed,
        'args': vars(args),
        'texts': TEXTS,
        'git_commit': git_commit(),
        'sv_model': SV_MODEL,
        'sv_revision': sv_revision(),
        'sha256': {
            'script': sha256(__file__),
            'checkpoint': sha256(args.checkpoint),
            'config': sha256(args.config),
            'voices': {s: sha256(voice_dir / f'{s}.pt') for s in speakers},
        },
        'top1': correct,
        'top1_real': real_correct,
        'median_rank': float(np.median(list(ranks.values()))),
        'mean_margin': float(np.mean(list(margins.values()))),
        'mean_margin_real': float(np.mean(list(real_margins.values()))),
        'between_generated': between(generated),
        'between_real': between(real),
        'between_generated_centered': between(generated, offset),
        'between_real_centered': between(real, offset),
        'between_real_same_text': between(same_text) if len(same_text) > 1 else None,
        'between_generated_same_text': between(generated_same_text) if len(generated_same_text) > 1 else None,
        'same_text_speakers': sorted(same_text),
        'generated_self_closest': gen_self,
        'generated_nearest': gen_nearest,
        'generated_matrix': {'speakers': speakers, 'cosine': G.round(4).tolist()},
        'ranks': ranks,
        'margins': margins,
        'margins_real': real_margins,
        'real_files': real_files,
    }
    print(f'\nSpeaker ID vs real centroids: top-1 {correct}/{n} (real recordings: {real_correct}/{n}), median rank {results["median_rank"]:g}/{n}')
    print(f'Mean margin: generated {results["mean_margin"]:.3f}, real {results["mean_margin_real"]:.3f} (higher = more identifiable)')
    print(f'Between-speaker cosine: generated {results["between_generated"]:.3f}, real {results["between_real"]:.3f} (lower = more distinct)')
    print(f'Centered between-speaker cosine: generated {results["between_generated_centered"]:.3f}, real {results["between_real_centered"]:.3f}')
    if results['between_real_same_text'] is not None:
        print(f'Same-text control ({len(same_text)} speakers): generated {results["between_generated_same_text"]:.3f}, real {results["between_real_same_text"]:.3f}')
    print(f'Generated voices closest to themselves across texts: {gen_self}/{n}')
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=1)
    print(f'Results: {out / "results.json"}')


if __name__ == '__main__':
    main()
