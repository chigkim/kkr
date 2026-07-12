#!/usr/bin/env python3
'''
Prepare a folder-organized multilingual dataset for StyleTTS2/Kokoro training.

Expected layout:
  dataset_root/
    en-us/
      speaker1/
        utt001.wav
        utt001.txt
      speaker2/
        utt001.wav
        utt001.txt
    de/
      speaker3/
        utt001.wav
        utt001.txt

Each .txt file contains the transcript for the .wav with the same stem.
Writes StyleTTS2 lists with phonemes already generated:
  wav_path|phonemes|speaker_id
'''

import argparse
import json
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_vctk_training import phonemize_by_language

_STYLE_ROOT = _REPO_ROOT / 'StyleTTS2'
if str(_STYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STYLE_ROOT))
from kokoro_symbols import symbols as KOKORO_SYMBOLS

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


def discover_rows(dataset_root: Path, extensions):
    rows = []
    for wav_path in sorted(dataset_root.rglob('*')):
        if not wav_path.is_file() or wav_path.suffix.lower() not in extensions:
            continue

        rel_parts = wav_path.relative_to(dataset_root).parts
        if len(rel_parts) < 3:
            continue

        language = rel_parts[0]
        speaker = rel_parts[1]
        text_path = wav_path.with_suffix('.txt')
        if not text_path.exists():
            print(f'WARNING: missing transcript for {wav_path}', file=sys.stderr)
            continue

        text = text_path.read_text(encoding='utf-8').strip()
        if not text:
            print(f'WARNING: empty transcript for {text_path}', file=sys.stderr)
            continue

        rows.append({
            'file': str(wav_path.relative_to(dataset_root)),
            'text': text,
            'speaker': f'{language}/{speaker}',
            'language': language,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Prepare multilingual lang/speaker wav+txt dataset for StyleTTS2 training')
    parser.add_argument('dataset', type=Path, help='Root with layout lang-code/speaker/*.wav + *.txt')
    parser.add_argument('-o', '--out-dir', type=Path, default=Path('training'), help='Processed dataset output directory')
    parser.add_argument('--val-ratio', type=float, default=VAL_RATIO)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--extensions', default='.wav', help='Comma-separated audio extensions, e.g. .wav,.flac')
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.out_dir = args.out_dir.resolve()

    extensions = {ext.strip().lower() for ext in args.extensions.split(',') if ext.strip()}
    rows = discover_rows(args.dataset, extensions)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f'No rows found under {args.dataset}. Expected lang-code/speaker/*.wav with matching .txt files.')

    speakers = {speaker: i for i, speaker in enumerate(sorted({r['speaker'] for r in rows}))}

    print(f'Found {len(rows):,} utterances')
    print(f'Languages: {", ".join(sorted({r["language"] for r in rows}))}')
    print(f'Speakers: {len(speakers):,}')
    print('Phonemizing with phonemizer/espeak...')
    phs = phonemize_by_language(rows, default_language=None, g2p_backend='phonemizer')

    entries = []
    for row, ipa in zip(rows, phs):
        ipa = ipa.strip()
        if len(ipa) < 5:
            continue
        entries.append((row['file'], ipa, speakers[row['speaker']], row['text'], row['language'], row['speaker']))

    train_entries, val_entries = split_by_speaker(entries, args.val_ratio, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / 'train.txt'
    val_path = args.out_dir / 'val.txt'
    ood_path = args.out_dir / 'OOD.txt'
    speaker_path = args.out_dir / 'speakers.json'
    symbols_path = args.out_dir / 'symbols.json'
    manifest_path = args.out_dir / 'manifest.json'

    def write_list(path, items):
        with open(path, 'w', encoding='utf-8') as f:
            for rel, ipa, speaker_id, _text, _language, _speaker in items:
                f.write(f'{rel}|{ipa}|{speaker_id}\n')

    write_list(train_path, train_entries)
    write_list(val_path, val_entries)

    with open(ood_path, 'w', encoding='utf-8') as f:
        for _rel, ipa, _speaker_id, _text, _language, _speaker in val_entries[:500]:
            f.write(ipa + '\n')

    with open(speaker_path, 'w', encoding='utf-8') as f:
        json.dump(speakers, f, indent=2, ensure_ascii=False)
        f.write('\n')

    base_symbols = list(KOKORO_SYMBOLS)
    base_set = set(base_symbols)
    symbol_counts = {}
    for _rel, ipa, _speaker_id, _text, _language, _speaker in entries:
        for symbol in ipa:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    added_symbols = sorted(symbol for symbol in symbol_counts if symbol not in base_set)
    expanded_symbols = base_symbols + added_symbols
    with open(symbols_path, 'w', encoding='utf-8') as f:
        json.dump({
            'base_vocab_size': len(base_symbols),
            'vocab_size': len(expanded_symbols),
            'symbols': expanded_symbols,
            'added_symbols': added_symbols,
            'added_symbol_counts': {symbol: symbol_counts[symbol] for symbol in added_symbols},
        }, f, indent=2, ensure_ascii=False)
        f.write('\n')

    # Emit an inference-ready Kokoro config with the expanded vocab so KModel
    # can load checkpoints trained on this dataset (base config has n_token=178).
    inference_config_path = args.out_dir / 'config.json'
    base_config_path = Path(__file__).resolve().parent / 'training' / 'config.json'
    if base_config_path.exists():
        with open(base_config_path, encoding='utf-8') as f:
            inference_config = json.load(f)
        inference_config['n_token'] = len(expanded_symbols)
        for index, symbol in enumerate(added_symbols, start=len(base_symbols)):
            inference_config['vocab'][symbol] = index
        with open(inference_config_path, 'w', encoding='utf-8') as f:
            json.dump(inference_config, f, indent=2, ensure_ascii=False)
            f.write('\n')
    else:
        print(f'WARNING: {base_config_path} not found; skipping inference config')
        inference_config_path = None

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': str(args.dataset),
            'processed_dataset': str(args.out_dir),
            'g2p_backend': 'phonemizer',
            'languages': sorted({r['language'] for r in rows}),
            'speakers': speakers,
            'num_entries': len(entries),
            'symbols': str(symbols_path),
            'inference_config': str(inference_config_path) if inference_config_path else None,
            'base_vocab_size': len(base_symbols),
            'vocab_size': len(expanded_symbols),
            'added_symbols': added_symbols,
            'split_strategy': 'per_speaker',
            'val_ratio': args.val_ratio,
            'seed': args.seed,
        }, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f'Valid entries : {len(entries):,}')
    print(f'Split         : per-speaker random val_ratio={args.val_ratio:g} seed={args.seed}')
    print(f'Train         : {len(train_entries):,} -> {train_path}')
    print(f'Val           : {len(val_entries):,} -> {val_path}')
    print(f'OOD           : {ood_path}')
    print(f'Speaker map   : {speaker_path}')
    print(f'Symbols       : {symbols_path} ({len(expanded_symbols)} symbols, +{len(added_symbols)})')
    if added_symbols:
        print(f'Added symbols : {" ".join(added_symbols)}')
    if inference_config_path:
        print(f'Infer config  : {inference_config_path} (pass as --config for Kokoro inference)')
    print(f'Manifest      : {manifest_path}')
    print(f'Set config data_params.root_path to: {args.dataset}')


if __name__ == '__main__':
    main()
