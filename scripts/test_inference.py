#!/usr/bin/env python3
"""
Kokoro German: Test Inference
==============================
Tests the fine-tuned Kokoro model with a German phonetic test set.

Usage:
    # Convert checkpoint + run inference
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --output-dir test_output/epoch3

    # Use a previously converted model
    python scripts/test_inference.py \
        --model voices/kokoro_german_epoch3.pth \
        --voicepack voices/dm_daniel_epoch3.pt

    # Run on CPU
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro_german/epoch_1st_00002.pth \
        --voicepack voices/dm_daniel_epoch3.pt \
        --device cpu
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Prefer the kokoro submodule over any pip-installed kokoro package
_repo_root = Path(__file__).resolve().parents[1]
_kokoro_submodule = _repo_root / "kokoro"
if _kokoro_submodule.exists() and str(_kokoro_submodule) not in sys.path:
    sys.path.insert(0, str(_kokoro_submodule))

# Standard German phonetic test set — covers all major pronunciation challenges
GERMAN_TEST_SENTENCES = [
    # 1. Umlauts (ä, ö, ü) and sch
    "Schön, dass du da bist. Die Bücher liegen auf dem großen Tisch.",
    # 2. Ich-Laut vs Ach-Laut (ç vs x)
    "Ich mache mich auf den Weg nach Aachen, um auch nachts wach zu sein.",
    # 3. Eszett (ß) and vowel length
    "Er aß die Maße in der Straße, aber das Maß war voll.",
    # 4. Zischlaute (z, ts) and consonant clusters
    "Zwei weiße Zwerge zwängen sich zwischen zwei Zweige.",
    # 5. Pf-Laute
    "Ein Pfau pflegt seine Federn an der Pfütze.",
    # 6. Prosody: questions and exclamations
    "Warum hast du das getan? Das ist ja unglaublich!",
    # 7. Numbers
    "Das kostet genau einhundertdreiundzwanzig Millionen Euro.",
]

# Standard American English test set
ENGLISH_TEST_SENTENCES = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The box was thrown beside the parked truck.",
    "The boy was amazed by the huge whale.",
]


def convert_checkpoint(checkpoint_path: str, output_path: str) -> str:
    """Convert a StyleTTS2 Stage 2 checkpoint to Kokoro KModel format.

    Extracts the 5 inference components (bert, bert_encoder, predictor,
    text_encoder, decoder) from the training checkpoint. All state dict
    keys must have the 'module.' prefix for KModel's loading fallback
    to work correctly.

    Requires that training was done with the new parametrizations API
    (torch.nn.utils.parametrizations.weight_norm/spectral_norm) so the
    state dict keys are natively compatible with Kokoro's KModel.
    """
    import torch

    print(f"Converting checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ckpt["net"]

    def ensure_module_prefix(state_dict):
        """Ensure all keys have 'module.' prefix for KModel compatibility."""
        return {
            ("module." + k if not k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    kokoro_weights = {}
    for key in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]:
        if key in net:
            kokoro_weights[key] = ensure_module_prefix(net[key])
            print(f"  {key}: {len(kokoro_weights[key])} keys")
        else:
            raise KeyError(f"'{key}' not found in checkpoint {checkpoint_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, str(output))
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Saved Kokoro-format weights: {output} ({size_mb:.1f} MB)")
    return str(output)


def run_inference(
    model_path: str,
    voicepack_path: str,
    config_path: str,
    output_dir: str,
    lang_code: str = "en-us",
    device: str = "auto",
    g2p_backend: str = "phonemizer",
):
    """Run inference on the test set."""
    import torch
    import soundfile as sf
    from kokoro import KModel, KPipeline

    if device == "auto":
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Language code: {lang_code}")

    # Load model with our fine-tuned weights and config
    print(f"Loading model from: {model_path}")
    print(f"  Config: {config_path}")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=config_path, model=model_path)
    kmodel = kmodel.to(device).eval()

    # Create pipeline with specified lang_code
    pipeline = KPipeline(
        lang_code=lang_code, repo_id="hexgrad/Kokoro-82M", model=kmodel, g2p_backend=g2p_backend
    )

    # Load voicepack
    print(f"Loading voicepack: {voicepack_path}")
    voice = torch.load(voicepack_path, map_location="cpu", weights_only=True)

    # Create output directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Select sentences based on lang_code
    test_sentences = (
        ENGLISH_TEST_SENTENCES
        if lang_code in ["a", "b"] or lang_code.startswith("en")
        else GERMAN_TEST_SENTENCES
    )

    # Generate audio for each test sentence
    print(f"\nGenerating {len(test_sentences)} test sentences...\n")
    for i, text in enumerate(test_sentences):
        print(f"[{i + 1}/{len(test_sentences)}] {text[:60]}...")
        try:
            generator = pipeline(text, voice=voice, speed=1)
            all_audio = []
            for gs, ps, audio in generator:
                print(f"  phonemes: {ps[:60]}...")
                all_audio.append(audio)

            if all_audio:
                import numpy as np

                combined = np.concatenate(all_audio)
                wav_path = out / f"test_{i + 1:02d}.wav"
                sf.write(str(wav_path), combined, 24000)
                duration = len(combined) / 24000
                print(f"  saved: {wav_path} ({duration:.1f}s)")
            else:
                print(f"  WARNING: No audio generated")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! Test audio saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned Kokoro model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--checkpoint",
        help="Path to StyleTTS2 checkpoint (.pth) — will be converted automatically",
    )
    group.add_argument(
        "--model",
        help="Path to already-converted Kokoro-format weights (.pth)",
    )
    parser.add_argument(
        "--voicepack",
        required=True,
        help="Path to voicepack (.pt)",
    )
    parser.add_argument(
        "--lang-code",
        default="en-us",
        help="Language code. With --g2p-backend phonemizer, an espeak language such as en-us, de, fr-fr; "
        "with kokoro, a Kokoro code such as a or d (default: en-us)",
    )
    parser.add_argument(
        "--g2p-backend",
        default="phonemizer",
        choices=["phonemizer", "kokoro"],
        help="Inference G2P; phonemizer matches lists from prepare_dataset.py (default: phonemizer)",
    )
    parser.add_argument(
        "--config",
        default="dataset/config.json",
        help="Path to Kokoro config.json; use the one prepare_dataset.py wrote so the vocab matches the checkpoint (default: dataset/config.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_output/",
        help="Directory to save generated WAV files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto)",
    )

    args = parser.parse_args()

    # Convert checkpoint if needed
    if args.checkpoint:
        model_path = convert_checkpoint(
            args.checkpoint,
            str(Path(args.output_dir) / "kokoro_converted.pth"),
        )
    else:
        model_path = args.model

    run_inference(
        model_path=model_path,
        voicepack_path=args.voicepack,
        config_path=args.config,
        output_dir=args.output_dir,
        lang_code=args.lang_code,
        device=args.device,
        g2p_backend=args.g2p_backend,
    )



if __name__ == "__main__":
    main()
