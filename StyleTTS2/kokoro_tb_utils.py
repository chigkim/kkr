"""Shared TensorBoard inference helpers for Kokoro-faithful audio previews.

Used by both train_first.py (Stage 1) and train_second.py (Stage 2) to generate
TensorBoard audio that matches actual Kokoro inference output — predicted duration,
F0, and energy from a voicepack, not ground-truth reconstruction.
"""

import logging
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

def prepare_test_tokens(text_cleaner, ood_path, n_sentences=8, min_chars=40, max_chars=200, seed=0):
    """Pick fixed held-out phoneme strings from the OOD list and tokenize them.

    The OOD list holds phonemes from the dataset's own frontend
    (prepare_dataset.py), so previews use the training language and
    phoneme conventions without running G2P here. The same lines are chosen
    every run (fixed seed).

    Returns a list of (display_text, token_ids) tuples; display_text is the
    phoneme string.
    """
    try:
        rows = [line.split('|') for line in Path(ood_path).read_text(encoding='utf-8').splitlines() if line.strip()]
        # Same format rule as meldataset: phonemes, or wav|phonemes|speaker
        idx = 1 if rows and '.wav' in rows[0][0] else 0
        lines = [row[idx].strip() for row in rows if len(row) > idx]
    except OSError as e:
        logger.warning(f"Could not read OOD texts for TensorBoard inference: {e}")
        return []
    lines = sorted({line for line in lines if min_chars <= len(line) <= max_chars})
    random.Random(seed).shuffle(lines)

    result = []
    for ipa in lines:
        token_ids = text_cleaner(ipa)
        if not token_ids or len(token_ids) > 510:
            continue
        result.append((ipa, token_ids))
        if len(result) >= n_sentences:
            break
    return result


def extract_voicepack(model, root_path, device, n_samples=200, speaker=None):
    """Extract a mini voicepack from audio files — mirrors extract_voicepack.py.

    Uses one speaker (WAV parent folder; default: the speaker with the most
    files) so the preview is a real voice, not an average across speakers.
    Samples up to n_samples of its files with a fixed seed, so every epoch
    uses the same references. Computes
    mel spectrograms with the same params as meldataset.py, runs them through
    model.style_encoder and model.predictor_encoder, and averages the results.

    Returns:
        voicepack: torch.FloatTensor [256] — combined acoustic+prosodic style
        acoustic_norm: float — output norm of style_encoder (health check)
        prosodic_norm: float — output norm of predictor_encoder (health check)
    """
    # sample_rate is deliberately omitted to match meldataset.py, which uses
    # torchaudio's default (16000) filterbank on 24 kHz audio.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)
    mel_mean, mel_std = -4, 4
    model = _unwrap_ddp(model)

    wav_files = list(Path(root_path).rglob("*.wav"))
    if not wav_files:
        logger.warning(f"extract_voicepack: no WAV files found in {root_path}")
        return None, 0.0, 0.0

    by_speaker = {}
    for wav_path in wav_files:
        by_speaker.setdefault(wav_path.parent.name, []).append(wav_path)
    if speaker is None:
        speaker = max(sorted(by_speaker), key=lambda k: len(by_speaker[k]))
    if speaker not in by_speaker:
        logger.warning(f"extract_voicepack: speaker {speaker} not found in {root_path}")
        return None, 0.0, 0.0
    wav_files = sorted(by_speaker[speaker])
    random.Random(0).shuffle(wav_files)
    wav_files = wav_files[:n_samples]
    logger.info(f"extract_voicepack: speaker {speaker}, {len(wav_files)} files")

    acoustic_styles = []
    prosodic_styles = []

    with torch.no_grad():
        for wav_path in wav_files:
            try:
                data, file_sr = sf.read(str(wav_path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data).unsqueeze(0)
                if file_sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, file_sr, 24000)
                # Same 5000-sample zero padding as meldataset.py's _load_tensor()
                waveform = torch.nn.functional.pad(waveform, (5000, 5000))
                waveform = waveform.to(device)
                mel = mel_transform(waveform)
                mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std
                if mel.shape[-1] < 80:
                    continue
                mel_input = mel.unsqueeze(1)  # [1, 1, 80, T]
                acoustic_styles.append(model.style_encoder(mel_input).cpu())
                prosodic_styles.append(model.predictor_encoder(mel_input).cpu())
            except Exception as e:
                logger.warning(f"extract_voicepack: skipping {wav_path.name}: {e}")

    if not acoustic_styles:
        logger.warning("extract_voicepack: no valid audio files processed")
        return None, 0.0, 0.0

    # Medoid, as in the default --style-mode of scripts/extract_voicepack.py,
    # so previews use the same kind of voicepack that evaluation does.
    avg_acoustic = _medoid(acoustic_styles)  # [128]
    avg_prosodic = _medoid(prosodic_styles)  # [128]
    voicepack = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]

    acoustic_norm = avg_acoustic.norm().item()
    prosodic_norm = avg_prosodic.norm().item()

    return voicepack.to(device), acoustic_norm, prosodic_norm


def _unwrap_ddp(model):
    # Plain modules for rank-0-only inference: no DDP attribute errors, no DDP collectives.
    return type(model)({k: getattr(v, 'module', v) for k, v in model.items()})


def _medoid(styles):
    """The per-utterance style closest to the mean of `styles` ([1, D] each)."""
    M = torch.cat(styles, dim=0)
    return M[(M - M.mean(dim=0)).norm(dim=1).argmin()]


def run_kokoro_inference(model, test_tokens, voicepack, device, text_cleaner):
    """Run Kokoro-faithful inference for all test sentences.

    Mirrors KModel.forward_with_tokens exactly:
      - Predict duration from the predictor
      - Build alignment from predicted duration
      - Predict F0 and energy
      - Decode audio using voicepack acoustic style

    Returns a list of (display_text, audio_numpy) tuples.
    """
    if voicepack is None or not test_tokens:
        return []
    model = _unwrap_ddp(model)

    # voicepack [256]: first 128 = acoustic (decoder), last 128 = prosodic (predictor)
    ref_acoustic = voicepack[:128].unsqueeze(0)  # [1, 128]
    ref_prosodic = voicepack[128:].unsqueeze(0)  # [1, 128]

    results = []
    with torch.no_grad():
        for text, token_ids in test_tokens:
            try:
                # Build input token tensor with BOS/EOS (token 0)
                input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
                input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
                text_mask = torch.gt(
                    torch.arange(input_lengths.max())
                    .unsqueeze(0)
                    .expand(1, -1)
                    .type_as(input_lengths)
                    + 1,
                    input_lengths.unsqueeze(1),
                ).to(device)

                # BERT + encoder
                bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                # Predict duration
                s_prosodic = ref_prosodic  # [1, 128]
                d = model.predictor.text_encoder(
                    d_en, s_prosodic, input_lengths, text_mask
                )
                x, _ = model.predictor.lstm(d)
                duration = model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze(0)).clamp(min=1).long()
                if pred_dur.dim() == 0:
                    pred_dur = pred_dur.unsqueeze(0)

                # Build alignment matrix from predicted duration
                n_tokens = input_ids.shape[1]
                total_frames = int(pred_dur.sum().item())
                if total_frames == 0:
                    continue
                pred_aln_trg = torch.zeros(n_tokens, total_frames).to(device)
                c_frame = 0
                for i in range(n_tokens):
                    dur_i = int(pred_dur[i].item())
                    pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
                    c_frame += dur_i
                pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [1, n_tokens, total_frames]

                # Predict F0 and energy
                en = d.transpose(-1, -2) @ pred_aln_trg
                F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosodic)

                # Text encoder + decode
                t_en = model.text_encoder(input_ids, input_lengths, text_mask)
                asr = t_en @ pred_aln_trg
                audio = model.decoder(asr, F0_pred, N_pred, ref_acoustic)
                results.append((text, audio.squeeze(1).cpu().numpy()))
            except Exception as e:
                logger.warning(f'Inference failed for "{text[:40]}": {e}')

    return results
