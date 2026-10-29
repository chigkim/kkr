#!/usr/bin/env python3
"""
Kokoro German: Extract Voicepack
=================================
Extracts a voicepack (.pt) from a fine-tuned StyleTTS2 checkpoint by running
both style encoders (acoustic + prosodic) on representative utterances and
averaging the resulting style vectors.

Usage:
    # Stage 2 checkpoint (both halves from the same checkpoint, whose decoder
    # was trained together with its style_encoder)
    python scripts/extract_voicepack.py \
        --model runs/af_punct/best_2nd.pth \
        --audio-dir af/en-us/p294 \
        --output voices/af_punct_stage2/p294.pt

    # Stage 1 checkpoint (predictor_encoder is untrained, so style_encoder is
    # used for both halves, as Stage 2 initializes it)
    python scripts/extract_voicepack.py \
        --model runs/af_punct/best_1st.pth \
        --audio-dir af/en-us/p294 \
        --output voices/af_punct_stage1/p294.pt \
        --device cpu

    # Hybrid for comparison: style_encoder from Stage 1, the rest from Stage 2
    python scripts/extract_voicepack.py \
        --model runs/af_punct/best_2nd.pth \
        --style-encoder-model runs/af_punct/best_1st.pth \
        --audio-dir af/en-us/p294 \
        --output voices/af_punct_hybrid/p294.pt

The files used are listed next to the voicepack in <name>.files.json, so
evaluation can keep them out of its real-recording enrollment set.

Voicepack format: tensor of shape [510, 1, 256] (float32)
  - 510 = max phoneme sequence length
  - 256 = style_dim * 2 (128 for decoder/timbre + 128 for predictor/prosody)
  - First 128 dims come from style_encoder (acoustic/timbre)
  - Last 128 dims come from predictor_encoder (prosody)
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# ── StyleTTS2 model components (standalone, no external imports) ─────────


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


# ── Main extraction logic ───────────────────────────────────────────────


def _predictor_encoder_trained(checkpoint):
    """Whether the checkpoint's predictor_encoder was ever optimized.

    Stage 1 never steps predictor_encoder, so its Adam state stays empty
    until Stage 2. Returns None when the checkpoint has no optimizer state.
    """
    optimizer_state = dict(checkpoint.get("optimizer") or [])
    if "predictor_encoder" not in optimizer_state:
        return None
    return len(optimizer_state["predictor_encoder"].get("state", {})) > 0


def extract_voicepack(
    model_path: str,
    audio_dir: str,
    output_path: str,
    num_samples: int = 200,
    device: str = "auto",
    style_encoder_model: str = None,
    style_mode: str = "medoid",
):
    import random

    import soundfile as sf
    import torchaudio

    # ── Resolve device ───────────────────────────────────────────────────
    if device == "auto":
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Collect audio files ──────────────────────────────────────────────
    audio_path = Path(audio_dir)
    wav_files = sorted(audio_path.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {audio_dir}")
        sys.exit(1)

    print(f"Found {len(wav_files):,} WAV files in {audio_dir}")

    # Sample a subset for embedding
    rng = random.Random(42)
    if len(wav_files) > num_samples:
        wav_files = rng.sample(wav_files, num_samples)
    print(f"Using {len(wav_files)} samples for voicepack extraction")

    # ── Build style encoders and load weights ────────────────────────────

    def strip_prefix(state_dict, prefix="module."):
        """Strip DataParallel 'module.' prefix from state dict keys if present."""
        if any(k.startswith(prefix) for k in state_dict.keys()):
            return {
                k[len(prefix) :] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
        return state_dict

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    net = checkpoint["net"]

    # Kokoro-82M architecture: dim_in=64, style_dim=128, hidden_dim=512
    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    predictor_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

    # Load style_encoder weights — from a separate checkpoint if provided,
    # otherwise from the main checkpoint. The default pairs the style_encoder
    # with the decoder it was trained with; the Stage 1 encoder with a Stage 2
    # decoder is an untested alternative, not a known improvement.
    if style_encoder_model:
        print(f"Loading style_encoder from separate checkpoint: {style_encoder_model}")
        se_checkpoint = torch.load(
            style_encoder_model, map_location="cpu", weights_only=False
        )
        se_net = se_checkpoint["net"]
        style_encoder.load_state_dict(strip_prefix(se_net["style_encoder"]))
        se_epoch = se_checkpoint.get("epoch", "?")
        print(f"  style_encoder loaded (epoch {se_epoch})")
    else:
        style_encoder.load_state_dict(strip_prefix(net["style_encoder"]))

    # predictor_encoder is only trained in Stage 2 (train_second.py); a
    # Stage 1 checkpoint holds its random initialization, whose outputs are
    # small and constant-looking rather than exploded, so output norms cannot
    # identify it. In that case use style_encoder for both halves of the
    # voicepack (the initialization Stage 2 uses — it deep-copies
    # style_encoder).
    predictor_encoder_trained = _predictor_encoder_trained(checkpoint)
    if predictor_encoder_trained is None:
        print("  WARNING: checkpoint has no optimizer state; assuming predictor_encoder is trained")
        predictor_encoder_trained = True
    if "predictor_encoder" in net:
        predictor_encoder.load_state_dict(strip_prefix(net["predictor_encoder"]))
    else:
        predictor_encoder_trained = False

    if not predictor_encoder_trained:
        print("  predictor_encoder appears untrained (Stage 1 only)")
        print("  Using style_encoder for both acoustic and prosodic embeddings")
        predictor_encoder.load_state_dict(
            strip_prefix(se_net["style_encoder"])
            if style_encoder_model
            else strip_prefix(net["style_encoder"])
        )

    style_encoder = style_encoder.to(device).eval()
    predictor_encoder = predictor_encoder.to(device).eval()

    # Validate style encoder output norms
    with torch.no_grad():
        dummy = torch.randn(1, 1, 80, 200).to(device)
        se_norm = style_encoder(dummy).norm().item()
        pe_norm = predictor_encoder(dummy).norm().item()
        print(f"  style_encoder output norm (dummy): {se_norm:.4f}")
        print(f"  predictor_encoder output norm (dummy): {pe_norm:.4f}")
        if se_norm < 0.5:
            print(f"  WARNING: style_encoder norm is very low ({se_norm:.4f}).")
            print(f"  This may indicate a collapsed encoder. Consider using")
            print(f"  --style-encoder-model to load from a Stage 1 checkpoint.")

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", 0)
    print(f"Loaded encoders from checkpoint")
    print(f"  Epoch: {epoch}, Val loss: {val_loss:.4f}")

    # ── Mel spectrogram transform ────────────────────────────────────────
    # Must match StyleTTS2's training preprocessing exactly:
    #   mel = to_mel(waveform)   # n_fft=2048, win=1200, hop=300, n_mels=80
    #   mel = (log(1e-5 + mel) - mean) / std   # mean=-4, std=4
    # sample_rate is deliberately omitted: meldataset.py uses torchaudio's default
    # (16000) filterbank on 24 kHz audio, and the encoders only work on that input.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)

    mel_mean = -4
    mel_std = 4

    # ── Extract style vectors ────────────────────────────────────────────
    acoustic_styles = []
    prosodic_styles = []
    used_files = []
    skipped = 0

    print(f"\nExtracting style vectors...")
    with torch.no_grad():
        for i, wav_path in enumerate(wav_files):
            # Load audio via soundfile (avoids torchcodec dependency)
            data, sr = sf.read(str(wav_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # mono
            if sr != 24000:
                # Resample using torchaudio
                waveform = torch.from_numpy(data).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            else:
                waveform = torch.from_numpy(data).unsqueeze(0)  # [1, samples]

            # Same 5000-sample zero padding as meldataset.py's _load_tensor()
            waveform = F.pad(waveform, (5000, 5000))
            waveform = waveform.to(device)

            # Compute mel spectrogram (same as StyleTTS2's preprocess())
            mel = mel_transform(waveform)  # [1, 80, T]
            mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std  # normalize

            # Skip clips with very short mels (style encoder needs >= 80 frames)
            if mel.shape[-1] < 80:
                skipped += 1
                continue

            mel_input = mel.unsqueeze(1)  # [1, 1, 80, T] — add channel dim

            # Run both style encoders
            s_acoustic = style_encoder(mel_input)  # [1, 128]
            s_prosodic = predictor_encoder(mel_input)  # [1, 128]

            if not (torch.isfinite(s_acoustic).all() and torch.isfinite(s_prosodic).all()):
                raise ValueError(f"Non-finite style vector for {wav_path}; the checkpoint is broken")

            acoustic_styles.append(s_acoustic.cpu())
            prosodic_styles.append(s_prosodic.cpu())
            used_files.append(str(wav_path.resolve()))

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(wav_files)} files...")

    if skipped > 0:
        print(f"  Skipped {skipped} files (too short)")

    print(f"  Extracted {len(acoustic_styles)} style vectors")

    if not acoustic_styles:
        print("ERROR: No style vectors extracted. Check audio files.")
        sys.exit(1)

    # ── Combine per-utterance styles and build voicepack ─────────────────
    # Training conditions on single-utterance styles, so 'medoid' keeps a
    # real, in-distribution style: the utterance closest to the speaker mean.
    # Plain averaging gives a vector with a smaller norm than any single
    # utterance. Which mode reproduces speakers best has not been measured
    # with the current (training-matched) mel filterbank. Each half is
    # combined separately, so medoid may pick different utterances for them.
    def combine(styles):
        M = torch.cat(styles, dim=0)  # [N, 128]
        mean = M.mean(dim=0)
        if style_mode == "mean":
            return mean
        if style_mode == "mean-renorm":
            return mean * (M.norm(dim=1).mean() / mean.norm().clamp(min=1e-8))
        if style_mode == "medoid":
            return M[(M - mean).norm(dim=1).argmin()]
        raise ValueError(f"unknown style_mode: {style_mode}")

    avg_acoustic = combine(acoustic_styles)  # [128]
    avg_prosodic = combine(prosodic_styles)  # [128]
    print(f"  Style mode: {style_mode}")

    # Kokoro voicepack format: [510, 1, 256]
    # First 128 dims = acoustic (decoder conditioning / timbre)
    # Last 128 dims = prosodic (predictor conditioning / prosody)
    combined = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]
    voicepack = combined.unsqueeze(0).unsqueeze(0).expand(510, 1, 256).clone()

    print(f"\nVoicepack shape: {tuple(voicepack.shape)}")
    print(f"  Acoustic style norm: {avg_acoustic.norm():.4f}")
    print(f"  Prosodic style norm: {avg_prosodic.norm():.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voicepack, str(output))
    print(f"\nSaved voicepack: {output} ({output.stat().st_size / 1024:.1f} KB)")
    files_path = output.with_suffix(".files.json")
    files_path.write_text(json.dumps(used_files, indent=1) + "\n", encoding="utf-8")
    print(f"Saved file list: {files_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract voicepack from fine-tuned Kokoro model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # After Stage 2 (both halves from the same checkpoint)
  python scripts/extract_voicepack.py \\
      --model runs/af_punct/best_2nd.pth \\
      --audio-dir af/en-us/p294 \\
      --output voices/af_punct_stage2/p294.pt

  # Stage 1 only (uses style_encoder for both halves), on CPU
  python scripts/extract_voicepack.py \\
      --model runs/af_punct/best_1st.pth \\
      --audio-dir af/en-us/p294 \\
      --output voices/af_punct_stage1/p294.pt \\
      --device cpu
""",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to fine-tuned StyleTTS2 checkpoint (.pth). Used for predictor_encoder "
        "(and style_encoder unless --style-encoder-model is specified).",
    )
    parser.add_argument(
        "--style-encoder-model",
        default=None,
        help="Path to a separate checkpoint for loading the style_encoder weights "
        "(default: --model). A Stage 1 checkpoint here with a Stage 2 --model gives "
        "the hybrid, which is an untested alternative to the default pairing.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory containing speaker's WAV files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output voicepack path (.pt)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of audio samples to use for extraction (default: 200)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on: auto (default), cpu, or cuda",
    )
    parser.add_argument(
        "--style-mode",
        default="medoid",
        choices=["medoid", "mean", "mean-renorm"],
        help="How to combine per-utterance styles: medoid = real utterance style "
        "closest to the speaker mean (default), mean = average, "
        "mean-renorm = average rescaled to typical utterance norm",
    )

    args = parser.parse_args()
    extract_voicepack(
        model_path=args.model,
        audio_dir=args.audio_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device,
        style_encoder_model=args.style_encoder_model,
        style_mode=args.style_mode,
    )


if __name__ == "__main__":
    main()
