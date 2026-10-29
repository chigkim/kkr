#!/usr/bin/env python3
'''
Prepare VCTK-style 24 kHz dataset.json for StyleTTS2/Kokoro training.

Expected input layout:
  ../voices/vckt-24k/dataset.json
  ../voices/vckt-24k/24k/<speaker>/<wav>.wav

Writes:
  training/vctk_train_list.txt
  training/vctk_val_list.txt
  training/vctk_OOD_texts.txt
  training/vctk_speakers.json
'''

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

VAL_RATIO = 0.05
RANDOM_SEED = 42


def split_by_speaker(entries, val_ratio, seed):
    '''Return train/val entries with validation sampled independently per speaker.'''
    rng = random.Random(seed)
    by_speaker = {}
    for entry in entries:
        speaker_id = entry[2]
        by_speaker.setdefault(speaker_id, []).append(entry)

    train_entries = []
    val_entries = []
    for speaker_id in sorted(by_speaker):
        speaker_entries = list(by_speaker[speaker_id])
        rng.shuffle(speaker_entries)
        if len(speaker_entries) <= 1:
            n_val = 0
        else:
            n_val = max(1, int(len(speaker_entries) * val_ratio))
            n_val = min(n_val, len(speaker_entries) - 1)
        val_entries.extend(speaker_entries[:n_val])
        train_entries.extend(speaker_entries[n_val:])

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def phonemize(texts, language, g2p_backend='phonemizer'):
    language = language.lower()

    if g2p_backend == 'phonemizer':
        try:
            from phonemizer import phonemize as phonemizer_phonemize
        except ImportError:
            print('ERROR: phonemizer is required. Run `uv sync`.', file=sys.stderr)
            raise

        phonemes = phonemizer_phonemize(
            texts,
            language=language,
            backend='espeak',
            strip=True,
            with_stress=True,
            preserve_punctuation=True,  # pauses and question intonation; base Kokoro was trained with punctuation
            punctuation_marks=';:,.!?—…"()“”',  # only marks in the base Kokoro vocab, so none become new tokens
            language_switch='remove-flags',  # espeak would otherwise insert (en)…(de) markers as tokens
            njobs=1,
        )
        return [' '.join(p.split()) for p in phonemes]

    try:
        from espeak_compat import configure_espeak
        configure_espeak()
        from misaki import en, espeak
    except ImportError:
        print('ERROR: misaki/espeak is required. Run `uv sync` and install espeak-ng.', file=sys.stderr)
        raise

    british = language in {'en-gb', 'british'}
    use_misaki_english = language in {'en', 'en-us', 'en-gb', 'american', 'british'}

    if g2p_backend == 'kokoro' and use_misaki_english:
        try:
            fallback = espeak.EspeakFallback(british=british)
        except Exception as e:
            print(f'WARNING: EspeakFallback not enabled; OOD words may be skipped: {e}', file=sys.stderr)
            fallback = None
        g2p = en.G2P(british=british, fallback=fallback, unk='')
    elif g2p_backend == 'misaki-espeak':
        g2p = espeak.EspeakG2P(language=language)
    else:
        raise ValueError(f'Unknown g2p backend: {g2p_backend}')

    phonemes = []
    for text in texts:
        ph, _ = g2p(text)
        phonemes.append(ph)
    return phonemes


def phonemize_by_language(rows, default_language, g2p_backend):
    phonemes = [None] * len(rows)
    groups = {}
    for index, row in enumerate(rows):
        language = row.get('language') or default_language
        groups.setdefault(language, []).append((index, row['text']))

    for language, items in sorted(groups.items()):
        print(f'  {language}: {len(items):,} texts')
        indices = [index for index, _text in items]
        texts = [text for _index, text in items]
        for index, ph in zip(indices, phonemize(texts, language, g2p_backend)):
            phonemes[index] = ph

    return phonemes


def main():
    parser = argparse.ArgumentParser(description='Prepare ../voices/vckt-24k for StyleTTS2 training')
    parser.add_argument('--dataset-root', type=Path, default=Path('../voices/vckt-24k'))
    parser.add_argument('--out-dir', type=Path, default=Path('training'))
    parser.add_argument('--language', default='en-us', help='G2P language, e.g. en-us, en-gb, de, fr-fr')
    parser.add_argument('--g2p-backend', default='phonemizer', choices=['phonemizer', 'kokoro', 'misaki-espeak'], help='phonemizer = espeak backend for all languages; kokoro = Kokoro/Misaki English; misaki-espeak = Misaki espeak wrapper')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--limit', type=int, default=None, help='Optional global cap for quick smoke tests')
    parser.add_argument('--speakers', default=None, help='Comma-separated speaker IDs to keep, e.g. p225,p226')
    parser.add_argument('--num-speakers', type=int, default=None, help='Keep the first N speakers alphabetically')
    parser.add_argument('--max-per-speaker', type=int, default=None, help='Keep at most N utterances per speaker')
    args = parser.parse_args()

    dataset_json = args.dataset_root / 'dataset.json'
    if not dataset_json.exists():
        raise FileNotFoundError(dataset_json)

    with open(dataset_json, encoding='utf-8') as f:
        rows = json.load(f)

    if args.speakers:
        keep_speakers = {s.strip() for s in args.speakers.split(',') if s.strip()}
        rows = [r for r in rows if r['speaker'] in keep_speakers]

    if args.num_speakers:
        keep_speakers = set(sorted({r['speaker'] for r in rows})[: args.num_speakers])
        rows = [r for r in rows if r['speaker'] in keep_speakers]

    if args.max_per_speaker:
        counts = {}
        limited_rows = []
        for row in rows:
            speaker = row['speaker']
            count = counts.get(speaker, 0)
            if count < args.max_per_speaker:
                limited_rows.append(row)
                counts[speaker] = count + 1
        rows = limited_rows

    if args.limit:
        rows = rows[: args.limit]

    missing = sum(1 for r in rows if not (args.dataset_root / r['file']).exists())
    rows = [r for r in rows if (args.dataset_root / r['file']).exists()]

    if not rows:
        raise ValueError('No dataset rows left after applying subset filters and checking wav files')

    speakers = {speaker: i for i, speaker in enumerate(sorted({r['speaker'] for r in rows}))}

    entries = []
    print(f'Phonemizing {len(rows):,} texts with backend={args.g2p_backend}...')
    phs = phonemize_by_language(rows, args.language, args.g2p_backend)

    for row, ipa in zip(rows, phs):
        rel = row['file']
        ipa = ipa.strip()
        if len(ipa) < 5:
            continue
        entries.append((rel, ipa, speakers[row['speaker']], row['text']))

    train_entries, val_entries = split_by_speaker(entries, args.val_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / 'vctk_train_list.txt'
    val_path = args.out_dir / 'vctk_val_list.txt'
    ood_path = args.out_dir / 'vctk_OOD_texts.txt'
    speaker_path = args.out_dir / 'vctk_speakers.json'

    def write_list(path, items):
        with open(path, 'w', encoding='utf-8') as f:
            for rel, ipa, speaker_id, _text in items:
                f.write(f'{rel}|{ipa}|{speaker_id}\n')

    write_list(train_path, train_entries)
    write_list(val_path, val_entries)

    # OOD file should contain phoneme strings; use held-out validation texts.
    with open(ood_path, 'w', encoding='utf-8') as f:
        for _rel, ipa, _speaker_id, _text in val_entries[:500]:
            f.write(ipa + '\n')

    with open(speaker_path, 'w', encoding='utf-8') as f:
        json.dump(speakers, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f'Valid entries : {len(entries):,}')
    print(f'Missing wavs  : {missing:,}')
    print(f'Speakers      : {len(speakers):,}')
    print(f'Split         : per-speaker random val_ratio={args.val_ratio:g} seed={args.seed}')
    print(f'Train         : {len(train_entries):,} -> {train_path}')
    print(f'Val           : {len(val_entries):,} -> {val_path}')
    print(f'OOD           : {ood_path}')
    print(f'Speaker map   : {speaker_path}')
    print('Set config data_params.root_path to the dataset root so wav paths resolve.')


if __name__ == '__main__':
    main()
