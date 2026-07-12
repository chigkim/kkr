import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parent
STYLETTS2_ROOT = REPO_ROOT / 'StyleTTS2'
if str(STYLETTS2_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLETTS2_ROOT))

import os.path as osp
import gc
import glob
import re
import sys
import yaml
import shutil
import numpy as np
import torch

if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )
import click
import warnings

warnings.simplefilter("ignore")

# load packages
import random
import yaml
from munch import Munch
import numpy as np
import torch

if getattr(torch, "_original_load", None) is None:
    torch._original_load = torch.load
    torch.load = lambda *args, **kwargs: torch._original_load(
        *args, **{**kwargs, "weights_only": False}
    )
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from meldataset import build_dataloader
from kokoro_symbols import TextCleaner
from symbols_config import apply_symbols_config
from kokoro_tb_utils import prepare_test_tokens, extract_voicepack, run_kokoro_inference
from speaker_losses import SpeakerClassifier, SpeakerCenters, speaker_separation_loss
from utils import *
from losses import *
from optimizers import build_optimizer
import time
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

import logging
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")

def _file_log(level, message, *args):
    if args:
        message = message % args
    target_logger = getattr(logger, 'logger', logger)
    record = target_logger.makeRecord(target_logger.name, level, __file__, 0, message, (), None)
    for handler in target_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)
    return message


def _print_and_log(level, message, *args):
    message = _file_log(level, message, *args)
    print(message)


def log_info(message, *args):
    _print_and_log(logging.INFO, message, *args)


def log_warning(message, *args):
    _print_and_log(logging.WARNING, message, *args)


def log_error(message, *args):
    _print_and_log(logging.ERROR, message, *args)


def scalar_value(value):
    if torch.is_tensor(value):
        return float(value.detach().mean().item())
    return float(value)


def find_latest_first_stage_checkpoint(log_dir):
    checkpoints = []
    pattern = osp.join(log_dir, "epoch_1st_*.pth")
    for path in glob.glob(pattern):
        match = re.search(r"epoch_1st_(\d+)\.pth$", osp.basename(path))
        if match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        return None

    return max(checkpoints, key=lambda item: item[0])


def atomic_torch_save(state, path):
    tmp_path = f"{path}.tmp"
    if osp.exists(tmp_path):
        os.remove(tmp_path)
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def prune_checkpoints(log_dir, pattern, keep_last, keep_paths=None):
    keep_paths = {osp.abspath(path) for path in (keep_paths or [])}
    if not keep_last or keep_last < 1:
        return
    checkpoints = []
    for path in glob.glob(osp.join(log_dir, pattern)):
        match = re.search(r"_(\d+)\.pth$", osp.basename(path))
        if match:
            checkpoints.append((int(match.group(1)), path))
    for _epoch, path in sorted(checkpoints)[:-keep_last]:
        if osp.abspath(path) not in keep_paths:
            os.remove(path)


def _resolve_path(path, base):
    if not path or osp.isabs(str(path)):
        return path
    return str((base / path).resolve())


def _apply_processed_dataset(config, dataset):
    if not dataset:
        return
    dataset_dir = Path(dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    dataset_dir = dataset_dir.resolve()

    data_params = config.setdefault('data_params', {})
    data_params['train_data'] = str(dataset_dir / 'train.txt')
    data_params['val_data'] = str(dataset_dir / 'val.txt')
    data_params['OOD_data'] = str(dataset_dir / 'OOD.txt')

    manifest_path = dataset_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path, encoding='utf-8') as f:
            manifest = yaml.safe_load(f)
        if manifest.get('dataset'):
            root_path = Path(manifest['dataset'])
            data_params['root_path'] = str(root_path if root_path.is_absolute() else (REPO_ROOT / root_path).resolve())

    symbols_path = dataset_dir / 'symbols.json'
    if symbols_path.exists():
        config.setdefault('text_params', {})['symbols_path'] = str(symbols_path)


def _apply_name(config, name, dataset):
    name = name or config.get('name')
    if not name:
        dataset_path = dataset or config.get('dataset')
        name = Path(dataset_path).name if dataset_path else 'default'
    run_dir = Path(name)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / 'runs' / run_dir
    config['log_dir'] = str(run_dir.resolve())


def _resolve_runtime_paths(config, config_path, dataset=None, name=None):
    dataset = dataset or config.get('dataset')
    _apply_processed_dataset(config, dataset)
    _apply_name(config, name, dataset)
    config_dir = Path(config_path).resolve().parent
    if config.get('pretrained_model'):
        config['pretrained_model'] = _resolve_path(config['pretrained_model'], REPO_ROOT)
    for key in ['ASR_config', 'ASR_path', 'F0_path', 'PLBERT_dir']:
        if key in config and config[key]:
            config[key] = _resolve_path(config[key], STYLETTS2_ROOT)
    data_params = config.get('data_params') or {}
    for key in ['train_data', 'val_data', 'root_path', 'OOD_data']:
        if key in data_params and data_params[key]:
            data_params[key] = _resolve_path(data_params[key], config_dir)


def _num_speakers_from_lists(*lists):
    speaker_ids = []
    for data_list in lists:
        for line in data_list:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                speaker_ids.append(int(parts[2]))
    return max(speaker_ids) + 1 if speaker_ids else 1


def _gb(value):
    return float(value) / 1024 ** 3


def _torch_memory_report_line(device):
    if device == 'mps' and torch.backends.mps.is_available():
        try:
            current = torch.mps.current_allocated_memory()
            driver = torch.mps.driver_allocated_memory()
            recommended = torch.mps.recommended_max_memory()
            return 'MPS: cur=%.2f drv=%.2f max=%.2f' % (_gb(current), _gb(driver), _gb(recommended))
        except Exception as exc:
            return f'MPS mem unavailable: {type(exc).__name__}'
    if device == 'cuda' and torch.cuda.is_available():
        try:
            return 'CUDA alloc=%.2f reserved=%.2f max=%.2f' % (
                _gb(torch.cuda.memory_allocated()),
                _gb(torch.cuda.memory_reserved()),
                _gb(torch.cuda.max_memory_allocated()),
            )
        except Exception as exc:
            return f'CUDA mem unavailable: {type(exc).__name__}'
    return 'torch_mem=n/a'


def _format_elapsed(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    return f'{minutes}m{seconds:02d}s'


def _format_eta(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours}h{minutes:02d}m{seconds:02d}s'


def _length_to_width_mask(lengths, width):
    '''Build a mask matching a padded/bucketed tensor width.'''
    mask = length_to_mask(lengths)
    if mask.size(1) < width:
        mask = F.pad(mask, (0, width - mask.size(1)), value=True)
    elif mask.size(1) > width:
        mask = mask[:, :width]
    return mask


def _length_to_text_mask(lengths, text_width):
    '''Build a mask matching padded/bucketed text tensor width.'''
    return _length_to_width_mask(lengths, text_width)


def _pad_last(x, width, value=0):
    if x.size(-1) == width:
        return x
    if x.size(-1) > width:
        return x[..., :width]
    return F.pad(x, (0, width - x.size(-1)), value=value)


def _time_gather(x, starts, length):
    '''Gather a fixed-width time crop with tensor starts, preserving gradients.'''
    base = torch.arange(length, device=x.device, dtype=torch.long).unsqueeze(0)
    idx = starts.to(x.device, dtype=torch.long).unsqueeze(1) + base
    while idx.ndim < x.ndim:
        idx = idx.unsqueeze(1)
    idx = idx.expand(*x.shape[:-1], length)
    return torch.gather(x, dim=-1, index=idx)


def _crop_adversarial_audio(y, y_hat, segment_size):
    '''Use a fixed waveform length for MPD/MSD/WavLM to avoid MPS graph-cache growth.'''
    if not segment_size or segment_size <= 0:
        return y, y_hat
    length = min(y.size(-1), y_hat.size(-1))
    if length > segment_size:
        start = torch.randint(0, length - segment_size + 1, (), device='cpu').item()
        return y[..., start:start + segment_size], y_hat[..., start:start + segment_size]
    if length < segment_size:
        pad = segment_size - length
        return F.pad(y[..., :length], (0, pad)), F.pad(y_hat[..., :length], (0, pad))
    return y[..., :length], y_hat[..., :length]


def _style_ref_crop(ref_mels, crop_frames, random_start=True):
    '''Fixed-width circular time crop for style/prosody encoder inputs.

    The collater zero-pads mels to the fixed MPS width, but the style encoders
    average over time without masking, so padding (which is not silence in
    normalized log-mel space) dilutes the style vector and mismatches
    inference, where styles come from unpadded mels.  A circular crop inside
    the true length keeps the shape fixed and never shows padding.  True
    lengths are recovered from the content: padded frames are exactly zero.
    '''
    if not crop_frames or crop_frames <= 0:
        return ref_mels
    crop_frames = int(crop_frames)
    lengths = (ref_mels.abs().sum(dim=1) > 0).sum(dim=-1).clamp(min=1)
    if random_start:
        starts = (torch.rand(ref_mels.size(0), device=ref_mels.device) * lengths).long()
    else:
        starts = torch.zeros_like(lengths)
    base = torch.arange(crop_frames, device=ref_mels.device).unsqueeze(0)
    idx = (starts.unsqueeze(1) + base) % lengths.unsqueeze(1)
    idx = idx.unsqueeze(1).expand(ref_mels.size(0), ref_mels.size(1), crop_frames)
    return torch.gather(ref_mels, dim=-1, index=idx)


def _s2s_loss(s2s_pred, texts, input_lengths):
    '''Fixed-shape aligner sequence-to-sequence CE loss.

    Equivalent to the original per-sample loop, but avoids variable-length
    slices on the training device: each distinct text length compiles a
    distinct MPS forward/backward graph that accumulates in driver memory.
    '''
    width = min(s2s_pred.size(1), texts.size(1))
    batch = s2s_pred.size(0)
    lengths = input_lengths.to(s2s_pred.device).clamp(max=width)
    ce = F.cross_entropy(
        s2s_pred[:, :width].reshape(-1, s2s_pred.size(-1)),
        texts[:, :width].reshape(-1),
        reduction='none',
    ).view(batch, width)
    pos = torch.arange(width, device=s2s_pred.device).unsqueeze(0)
    valid = (pos < lengths.unsqueeze(1)).to(ce.dtype)
    per_sample = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return per_sample.sum() / batch


@click.command()
@click.option("-p", "--config_path", default="Configs/config.yml", type=str)
@click.option('--dataset', type=str, default=None, help='Processed dataset directory from prepare_dataset.py')
@click.option('-name', '-n', '--name', type=str, default=None, help='Run/output name; saves checkpoints and logs under runs/<name>')
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from the latest epoch_1st_*.pth checkpoint in log_dir.",
)
def main(config_path, dataset, name, resume):
    config_path = str((REPO_ROOT / config_path).resolve()) if not osp.isabs(config_path) else config_path
    config = yaml.safe_load(open(config_path))
    _resolve_runtime_paths(config, config_path, dataset, name)
    symbols_config = apply_symbols_config(config, config_path)

    log_dir = config["log_dir"]
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    )
    logger.logger.addHandler(file_handler)
    if symbols_config:
        log_info(f'Using symbols: {symbols_config[0]} ({symbols_config[1]} tokens)')

    batch_size = config.get("batch_size", 10)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    log_info(f"Using {device} device")
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        log_info(f"Set torch CPU threads: {prev_threads} -> {torch.get_num_threads()} for MPS native-heap pressure")
        if bool(config.get('mps_native_lstm', False)):
            set_mps_native_lstm(True)
            log_info('Using native MPS packed LSTMs (mps_native_lstm=true)')
    log_info(f"Using device: {device}")

    epochs = config.get("epochs_1st", 200)
    save_freq = config.get("save_freq", 1)
    log_interval = config.get("log_interval", 10)
    saving_epoch = config.get("save_freq", 1)

    data_params = config.get("data_params", None)
    sr = config["preprocess_params"].get("sr", 24000)
    train_path = data_params["train_data"]
    val_path = data_params["val_data"]
    root_path = data_params["root_path"]
    min_length = data_params["min_length"]
    OOD_data = data_params["OOD_data"]
    num_workers = data_params.get("num_workers", 2)

    max_len = config.get("max_len", 200)

    # MPS fixed-shape mitigations (same knobs as Stage 2): variable widths
    # compile distinct MPS graph variants that accumulate in driver memory.
    adversarial_segment_size = int(config.get("adversarial_segment_size", sr if device == "mps" else 0) or 0)
    if adversarial_segment_size > 0:
        log_info(f"Using fixed MPD/MSD segment size: {adversarial_segment_size} samples")
    slm_segment_size = int(config.get("slm_segment_size", sr if device == "mps" else 0) or 0)
    if slm_segment_size > 0:
        log_info(f"Using fixed WavLM segment size: {slm_segment_size} samples")
    fixed_training_mel_len = bool(config.get('fixed_training_mel_len', device == 'mps'))
    fixed_training_mel_len_frames = int(config.get('fixed_training_mel_len_frames') or (max_len // 2))
    fixed_training_mel_len_frames = min(fixed_training_mel_len_frames, max_len // 2)
    if fixed_training_mel_len:
        log_info(f"Using fixed Stage 1 training mel length: {fixed_training_mel_len_frames} frames")
    fixed_mps_text_length = int(config.get('fixed_mps_text_length', 0) or 0)
    fixed_mps_mel_length = int(config.get('fixed_mps_mel_length', 0) or 0)
    mps_safe_gather_crop = bool(config.get('mps_safe_gather_crop', device == 'mps'))
    style_ref_crop_frames = int(config.get('style_ref_crop_frames', 240) or 0)
    if style_ref_crop_frames > 0:
        log_info(f"Using fixed style reference crop: {style_ref_crop_frames} frames")
    collate_config = {}
    if device == 'mps':
        if fixed_mps_text_length > 0:
            collate_config['fixed_text_length'] = fixed_mps_text_length
            log_info(f"Using fixed MPS text pad length: {fixed_mps_text_length}")
        if fixed_mps_mel_length > 0:
            collate_config['fixed_mel_length'] = fixed_mps_mel_length
            log_info(f"Using fixed MPS mel pad length: {fixed_mps_mel_length}")
        if mps_safe_gather_crop:
            log_info('Using tensor-gather crops for MPS-safe decoder windows')

    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(
        train_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_config={},
        collate_config=collate_config,
        device=device,
    )

    val_dataloader = build_dataloader(
        val_list,
        root_path,
        OOD_data=OOD_data,
        min_length=min_length,
        batch_size=batch_size,
        validation=True,
        num_workers=0,
        device=device,
        dataset_config={},
        collate_config=collate_config,
    )

    with accelerator.main_process_first():
        # load pretrained ASR model
        ASR_config = config.get("ASR_config", False)
        ASR_path = config.get("ASR_path", False)
        text_aligner = load_ASR_models(ASR_path, ASR_config, vocab_size=config['model_params']['n_token'])

        # load pretrained F0 model
        F0_path = config.get("F0_path", False)
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model
        from Utils.PLBERT.util import load_plbert

        BERT_path = config.get("PLBERT_dir", False)
        plbert = load_plbert(BERT_path, vocab_size=config['model_params']['n_token'])

    scheduler_params = {
        "max_lr": float(config["optimizer_params"].get("lr", 1e-4)),
        "pct_start": float(config["optimizer_params"].get("pct_start", 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    loss_params = Munch(config["loss_params"])
    diffusion_enabled = loss_params.get('lambda_diff', 0) > 0 and loss_params.get('diff_epoch', 999) < epochs
    model = build_model(model_params, text_aligner, pitch_extractor, plbert, enable_diffusion=diffusion_enabled)
    if not diffusion_enabled:
        log_info('Diffusion model disabled; skipping diffusion module allocation')

    num_speakers = int(config.get('num_speakers') or _num_speakers_from_lists(train_list, val_list))
    style_dim = int(config['model_params'].get('style_dim', 128))
    if loss_params.get('lambda_timbre_spk', 0) > 0:
        model['timbre_speaker_classifier'] = SpeakerClassifier(style_dim, num_speakers)
    if loss_params.get('lambda_timbre_center', 0) > 0 or loss_params.get('lambda_speaker_margin', 0) > 0:
        model['timbre_speaker_centers'] = SpeakerCenters(num_speakers, style_dim)
    if 'timbre_speaker_classifier' in model or 'timbre_speaker_centers' in model:
        log_info(f'Enabled timbre speaker separation loss for {num_speakers} speakers')

    best_loss = float("inf")  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])

    TMA_epoch = loss_params.TMA_epoch

    for k in model:
        model[k] = accelerator.prepare(model[k])

    train_dataloader, val_dataloader = accelerator.prepare(
        train_dataloader, val_dataloader
    )

    _ = [model[key].to(device) for key in model]

    # initialize optimizers after preparing models for compatibility with FSDP
    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict={key: scheduler_params.copy() for key in model},
        lr=float(config["optimizer_params"].get("lr", 1e-4)),
    )

    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    resume_checkpoint = None
    if resume:
        latest_checkpoint = find_latest_first_stage_checkpoint(log_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"--resume was set, but no epoch_1st_*.pth checkpoints were found in {log_dir}"
            )
        resume_epoch, resume_checkpoint = latest_checkpoint
        log_info(
            f"Resuming from latest first-stage checkpoint: {resume_checkpoint} "
            f"(epoch {resume_epoch})"
        )

    with accelerator.main_process_first():
        if resume_checkpoint is not None or config.get("pretrained_model", "") != "":
            checkpoint_path = resume_checkpoint or config["pretrained_model"]
            load_only_params = False if resume_checkpoint is not None else config.get(
                "load_only_params", True
            )
            model, optimizer, start_epoch, iters = load_checkpoint(
                model,
                optimizer,
                checkpoint_path,
                load_only_params=load_only_params,
            )
            if not load_only_params:
                best_path = osp.join(log_dir, config.get("best_first_stage_path", "best_1st.pth"))
                best_source = best_path if osp.exists(best_path) else checkpoint_path
                try:
                    best_value = torch.load(best_source, map_location="cpu").get("val_loss", best_loss)
                    best_loss = float(best_value.item() if torch.is_tensor(best_value) else best_value)
                    log_info("Loaded best Stage 1 validation loss %.6f from %s", best_loss, best_source)
                except Exception:
                    log_warning("Could not load best Stage 1 validation loss from %s", best_source)
                start_epoch += 1
                log_info(f"Resuming first-stage training from epoch {start_epoch}/{epochs - 1}")
        else:
            start_epoch = 0
            iters = 0

    if start_epoch >= epochs:
        if accelerator.is_main_process:
            log_info(
                f"No Stage 1 epochs left to run: next epoch would be {start_epoch}/{epochs - 1}, "
                f"but epochs_1st is {epochs}. Increase epochs_1st in the config to continue."
            )
        return

    # in case not distributed
    try:
        n_down = model.text_aligner.module.n_down
    except:
        n_down = model.text_aligner.n_down

    # wrapped losses for compatibility with mixed precision
    stft_loss = MultiResolutionSTFTLoss().to(device)
    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(device)

    # ── Kokoro-faithful TensorBoard inference setup ───────────────────────────
    if accelerator.is_main_process:
        text_cleaner = TextCleaner()
        _test_tokens = prepare_test_tokens(text_cleaner)
        if _test_tokens:
            log_info(
                f"TensorBoard inference: prepared {len(_test_tokens)} test sentences"
            )
        else:
            log_warning(
                "TensorBoard inference: no test sentences available (misaki/espeak not found?)"
            )

        # Generate pretrained baseline before any training updates
        baseline_samples = config.get("tensorboard_baseline_samples", 200)
        if baseline_samples > 0:
            _ = [model[key].eval() for key in model]
            log_info("Extracting pretrained baseline voicepack for TensorBoard...")
            _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (
                extract_voicepack(model, root_path, device, n_samples=baseline_samples)
            )
            if _baseline_voicepack is not None:
                writer.add_scalar("baseline/acoustic_norm", _baseline_acoustic_norm, 0)
                writer.add_scalar("baseline/prosodic_norm", _baseline_prosodic_norm, 0)
                log_info(
                    f"  acoustic_norm={_baseline_acoustic_norm:.4f}  prosodic_norm={_baseline_prosodic_norm:.4f}"
                )
                _baseline_audio = run_kokoro_inference(
                    model, _test_tokens, _baseline_voicepack, device, text_cleaner
                )
                for i, (text, audio) in enumerate(_baseline_audio):
                    writer.add_audio(f"baseline/test_{i + 1:02d}", audio, 0, sample_rate=sr)
                    log_info(f"  baseline/test_{i + 1:02d}: {text[:60]}")
            _ = [model[key].train() for key in model]
        else:
            log_info("Skipping TensorBoard baseline voicepack extraction")
    # ─────────────────────────────────────────────────────────────────────────

    epoch_durations = []
    for epoch in range(start_epoch, epochs):
        if device == "mps":
            gc.collect()
            torch.mps.empty_cache()
        running_loss = 0
        skipped_short_batches = 0
        dropped_short_samples = 0
        start_time = time.time()

        _ = [model[key].train() for key in model]

        train_progress = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc='Epoch %d/%d' % (epoch, epochs - 1),
            unit='step',
            dynamic_ncols=True,
            disable=not accelerator.is_main_process,
        )
        for i, batch in train_progress:
            waves = batch[0]
            labels = batch[-2].to(device)
            ref_labels = batch[-1].to(device)
            batch = [b.to(device) for b in batch[1:-2]]
            texts, input_lengths, _, _, mels, mel_input_length, ref_mels = batch

            if fixed_training_mel_len:
                # Drop only the samples too short for the fixed crop window
                # instead of skipping the whole batch. Batch-size variants are
                # bounded by batch_size, so they cannot leak MPS graph shapes.
                keep = (mel_input_length // 2) > (fixed_training_mel_len_frames + 1)
                if not bool(keep.all()):
                    if not bool(keep.any()):
                        skipped_short_batches += 1
                        continue
                    keep_idx = keep.nonzero(as_tuple=True)[0]
                    dropped_short_samples += len(waves) - keep_idx.numel()
                    waves = [waves[k] for k in keep_idx.tolist()]
                    labels = labels[keep_idx]
                    ref_labels = ref_labels[keep_idx]
                    texts = texts[keep_idx]
                    input_lengths = input_lengths[keep_idx]
                    mels = mels[keep_idx]
                    mel_input_length = mel_input_length[keep_idx]
                    ref_mels = ref_mels[keep_idx]

            with torch.no_grad():
                aligner_width = mels.size(2) // (2**n_down)
                mask = _length_to_width_mask(mel_input_length // (2**n_down), aligner_width).to(device)
                text_mask = _length_to_text_mask(input_lengths, texts.size(1)).to(texts.device)


            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)

            with torch.no_grad():
                attn_mask = (
                    (~mask)
                    .unsqueeze(-1)
                    .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                    .float()
                    .transpose(-1, -2)
                )
                attn_mask = (
                    attn_mask.float()
                    * (~text_mask)
                    .unsqueeze(-1)
                    .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                    .float()
                )
                attn_mask = attn_mask < 1

            s2s_attn.masked_fill_(attn_mask, 0.0)

            with torch.no_grad():
                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)

            # 50% of chance of using monotonic version
            if bool(random.getrandbits(1)):
                asr = t_en @ s2s_attn
            else:
                asr = t_en @ s2s_attn_mono

            # get clips
            if fixed_training_mel_len:
                # Pin decoder/STFT/WavLM/MPD/MSD shapes on MPS. Variable mel_len
                # creates distinct MPS graph variants that accumulate in
                # driver/native memory.
                # Too-short samples were already dropped at the top of the loop.
                mel_len = fixed_training_mel_len_frames
            else:
                mel_input_length_all = accelerator.gather(
                    mel_input_length
                )  # for balanced load
                mel_len = min(
                    [int(mel_input_length_all.min().item() / 2 - 1), max_len // 2]
                )

            starts = []
            wav = []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                starts.append(int(random_start))
                y = waves[bib][
                    (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                ]
                wav.append(torch.from_numpy(y).float().to(device))

            wav = torch.stack(wav).float().detach()

            if mps_safe_gather_crop:
                starts_t = torch.as_tensor(starts, device=device, dtype=torch.long)
                en = _time_gather(asr, starts_t, mel_len)
                gt = _time_gather(mels, starts_t * 2, mel_len * 2).detach()
            else:
                en = torch.stack([asr[bib, :, s0 : s0 + mel_len] for bib, s0 in enumerate(starts)])
                gt = torch.stack([
                    mels[bib, :, (s0 * 2) : ((s0 + mel_len) * 2)] for bib, s0 in enumerate(starts)
                ]).detach()
            # Use actual reference from a different clip; the circular crop is
            # padding-free and fixed-shape (see _style_ref_crop).
            st = _style_ref_crop(ref_mels.detach(), style_ref_crop_frames)

            # clip too short to be used by the style encoder
            if gt.shape[-1] < 80:
                continue

            with torch.no_grad():
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1).detach()
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))

            s = model.style_encoder(
                st.unsqueeze(1) if multispeaker else gt.unsqueeze(1)
            )
            timbre_spk_loss, timbre_spk_ce, timbre_center_loss, timbre_margin_loss = speaker_separation_loss(
                s,
                ref_labels if multispeaker else labels,
                classifier=model.get('timbre_speaker_classifier'),
                centers=model.get('timbre_speaker_centers'),
                lambda_ce=loss_params.get('lambda_timbre_spk', 0),
                lambda_center=loss_params.get('lambda_timbre_center', 0),
                lambda_margin=loss_params.get('lambda_speaker_margin', 0),
                margin=loss_params.get('speaker_center_margin', 1.0),
            )

            y_rec = model.decoder(en, F0_real, real_norm, s)

            # discriminator loss

            wav_g = wav.unsqueeze(1).float()
            if epoch >= TMA_epoch:
                wav_adv, y_rec_adv = _crop_adversarial_audio(wav_g, y_rec, adversarial_segment_size)
                optimizer.zero_grad()
                d_loss = dl(wav_adv.detach(), y_rec_adv.detach()).mean()
                accelerator.backward(d_loss)
                optimizer.step("msd")
                optimizer.step("mpd")
            else:
                d_loss = 0

            # generator loss
            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            if epoch >= TMA_epoch:  # start TMA training
                loss_s2s = _s2s_loss(s2s_pred, texts, input_lengths)

                loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

                loss_gen_all = gl(wav_adv.detach(), y_rec_adv).mean()
                wav_lm, y_rec_lm = _crop_adversarial_audio(wav_g.detach(), y_rec, slm_segment_size)
                loss_slm = wl(wav_lm.squeeze(1), y_rec_lm.squeeze(1)).mean()

                g_loss = (
                    loss_params.lambda_mel * loss_mel
                    + loss_params.lambda_mono * loss_mono
                    + loss_params.lambda_s2s * loss_s2s
                    + loss_params.lambda_gen * loss_gen_all
                    + loss_params.lambda_slm * loss_slm
                    + timbre_spk_loss
                )

            else:
                loss_s2s = 0
                loss_mono = 0
                loss_gen_all = 0
                loss_slm = 0
                g_loss = loss_mel + timbre_spk_loss

            running_loss += accelerator.gather(loss_mel).mean().item()

            accelerator.backward(g_loss)

            optimizer.step("text_encoder")
            optimizer.step("style_encoder")
            optimizer.step("decoder")
            if 'timbre_speaker_classifier' in model:
                optimizer.step("timbre_speaker_classifier")
            if 'timbre_speaker_centers' in model:
                optimizer.step("timbre_speaker_centers")

            if epoch >= TMA_epoch:
                optimizer.step("text_aligner")
                optimizer.step("pitch_extractor")

            mps_cache_interval = config.get("mps_empty_cache_interval", 25)
            if (
                device == "mps"
                and mps_cache_interval > 0
                and (i + 1) % mps_cache_interval == 0
            ):
                gc.collect()
                torch.mps.empty_cache()

            iters = iters + 1

            if accelerator.is_main_process:
                train_progress.set_postfix(
                    {
                        'Mel': '%.2f' % scalar_value(loss_mel),
                        'Gen': '%.2f' % scalar_value(loss_gen_all),
                        'Disc': '%.2f' % scalar_value(d_loss),
                        'Mono': '%.2f' % scalar_value(loss_mono),
                        'S2S': '%.2f' % scalar_value(loss_s2s),
                        'SLM': '%.2f' % scalar_value(loss_slm),
                        'TimbreCE': '%.2f' % scalar_value(timbre_spk_ce),
                        'TimbreCenter': '%.2f' % scalar_value(timbre_center_loss),
                        'SpkMargin': '%.2f' % scalar_value(timbre_margin_loss),
                        'MPS': _torch_memory_report_line(device).replace('MPS: ', ''),
                    },
                    refresh=True,
                )

            if (i + 1) % log_interval == 0 and accelerator.is_main_process:
                avg_mel_loss = running_loss / log_interval
                elapsed = _format_elapsed(time.time() - start_time)
                memory_line = _torch_memory_report_line(device)
                step_message = (
                    "Step %d/%d: Mel=%.2f Gen=%.2f Disc=%.2f Mono=%.2f S2S=%.2f SLM=%.2f TimbreCE=%.2f TimbreCenter=%.2f SpkMargin=%.2f Elapsed=%s\n%s"
                    % (
                        i + 1,
                        len(train_dataloader),
                        avg_mel_loss,
                        loss_gen_all,
                        d_loss,
                        loss_mono,
                        loss_s2s,
                        loss_slm,
                        timbre_spk_ce,
                        timbre_center_loss,
                        timbre_margin_loss,
                        elapsed,
                        memory_line,
                    )
                )
                _file_log(logging.INFO, step_message)
                train_progress.set_postfix(
                    {
                        'Mel': '%.2f' % avg_mel_loss,
                        'Gen': '%.2f' % scalar_value(loss_gen_all),
                        'Disc': '%.2f' % scalar_value(d_loss),
                        'Mono': '%.2f' % scalar_value(loss_mono),
                        'S2S': '%.2f' % scalar_value(loss_s2s),
                        'SLM': '%.2f' % scalar_value(loss_slm),
                        'TimbreCE': '%.2f' % scalar_value(timbre_spk_ce),
                        'TimbreCenter': '%.2f' % scalar_value(timbre_center_loss),
                        'SpkMargin': '%.2f' % scalar_value(timbre_margin_loss),
                        'MPS': memory_line.replace('MPS: ', ''),
                    },
                    refresh=True,
                )

                writer.add_scalar("train/mel_loss", avg_mel_loss, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/mono_loss", loss_mono, iters)
                writer.add_scalar("train/s2s_loss", loss_s2s, iters)
                writer.add_scalar("train/slm_loss", loss_slm, iters)
                writer.add_scalar("train/timbre_spk_ce", timbre_spk_ce, iters)
                writer.add_scalar("train/timbre_center_loss", timbre_center_loss, iters)
                writer.add_scalar("train/speaker_margin_loss", timbre_margin_loss, iters)

                running_loss = 0

        if (skipped_short_batches or dropped_short_samples) and accelerator.is_main_process:
            log_info(
                'Epoch=%d/%d skipped_short_batches=%d dropped_short_samples=%d fixed_mel_len=%d'
                % (epoch, epochs - 1, skipped_short_batches, dropped_short_samples, fixed_training_mel_len_frames)
            )

        loss_test = 0

        _ = [model[key].eval() for key in model]

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:-2]]
                texts, input_lengths, _, _, mels, mel_input_length, ref_mels = batch

                if fixed_training_mel_len:
                    # Same per-sample short-clip handling as training.
                    keep = (mel_input_length // 2) > (fixed_training_mel_len_frames + 1)
                    if not bool(keep.any()):
                        continue
                    if not bool(keep.all()):
                        keep_idx = keep.nonzero(as_tuple=True)[0]
                        waves = [waves[k] for k in keep_idx.tolist()]
                        texts = texts[keep_idx]
                        input_lengths = input_lengths[keep_idx]
                        mels = mels[keep_idx]
                        mel_input_length = mel_input_length[keep_idx]
                        ref_mels = ref_mels[keep_idx]

                with torch.no_grad():
                    aligner_width = mels.size(2) // (2**n_down)
                    mask = _length_to_width_mask(mel_input_length // (2**n_down), aligner_width).to(device)
                    ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2)

                    text_mask = _length_to_text_mask(input_lengths, texts.size(1)).to(texts.device)
                    attn_mask = (
                        (~mask)
                        .unsqueeze(-1)
                        .expand(mask.shape[0], mask.shape[1], text_mask.shape[-1])
                        .float()
                        .transpose(-1, -2)
                    )
                    attn_mask = (
                        attn_mask.float()
                        * (~text_mask)
                        .unsqueeze(-1)
                        .expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1])
                        .float()
                    )
                    attn_mask = attn_mask < 1
                    s2s_attn.masked_fill_(attn_mask, 0.0)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)

                asr = t_en @ s2s_attn

                # get clips
                if fixed_training_mel_len:
                    # Keep validation crops fixed-width too; validation runs on
                    # the training device in Stage 1. Too-short samples were
                    # already dropped after unpacking.
                    mel_len = fixed_training_mel_len_frames
                else:
                    mel_len = min(
                        [int(mel_input_length.min().item() / 2 - 1), max_len // 2]
                    )

                en = []
                gt = []
                wav = []
                for bib in range(len(mel_input_length)):
                    mel_length = int(mel_input_length[bib].item() / 2)

                    random_start = np.random.randint(0, mel_length - mel_len)
                    en.append(asr[bib, :, random_start : random_start + mel_len])
                    gt.append(
                        mels[
                            bib, :, (random_start * 2) : ((random_start + mel_len) * 2)
                        ]
                    )
                    y = waves[bib][
                        (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                    ]
                    wav.append(torch.from_numpy(y).float().to(device))

                wav = torch.stack(wav).float().detach()

                en = torch.stack(en)
                gt = torch.stack(gt).detach()

                F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))
                s = model.style_encoder(
                    _style_ref_crop(ref_mels.detach(), style_ref_crop_frames, random_start=False).unsqueeze(1)
                    if multispeaker
                    else gt.unsqueeze(1)
                )
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                y_rec = model.decoder(en, F0_real, real_norm, s)

                loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                loss_test += accelerator.gather(loss_mel).mean().item()
                iters_test += 1

        if accelerator.is_main_process:
            epoch_duration = time.time() - start_time
            epoch_durations.append(epoch_duration)
            remaining_epochs = max(epochs - (epoch + 1), 0)
            eta = sum(epoch_durations) / len(epoch_durations) * remaining_epochs
            if iters_test == 0:
                log_error('Validation produced no successful batches; keeping training alive with val_loss=inf')
                loss_test = float('inf')
                iters_test = 1
            log_info(
                "ETA=%s Validation loss=%.6f"
                % (_format_eta(eta), loss_test / iters_test)
            )
            writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch)
            attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
            writer.add_figure("eval/attn", attn_image, epoch)

            val_loss = loss_test / iters_test
            is_best = val_loss < best_loss
            should_save_epoch = epoch % saving_epoch == 0
            if is_best or should_save_epoch:
                state = {
                    "net": {key: model[key].state_dict() for key in model},
                    "optimizer": optimizer.state_dict(),
                    "iters": iters,
                    "val_loss": val_loss,
                    "epoch": epoch,
                }
                best_path = osp.join(log_dir, config.get("best_first_stage_path", "best_1st.pth"))
                save_path = None
                if should_save_epoch:
                    save_path = osp.join(log_dir, "epoch_1st_%05d.pth" % epoch)
                    atomic_torch_save(state, save_path)
                if is_best:
                    best_loss = val_loss
                    atomic_torch_save(state, best_path)
                    log_info("New best Stage 1 validation loss: %.6f at epoch %d" % (best_loss, epoch))
                keep_paths = [best_path]
                if save_path:
                    keep_paths.append(save_path)
                prune_checkpoints(
                    log_dir,
                    "epoch_1st_*.pth",
                    config.get("keep_last_checkpoints", 2),
                    keep_paths=keep_paths,
                )

            # ── Kokoro-faithful TensorBoard inference (every epoch) ───────────
            _ = [model[key].eval() for key in model]
            _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(
                model,
                root_path,
                device,
                n_samples=200,
            )
            writer.add_scalar("voicepack/acoustic_norm", _acoustic_norm, epoch)
            writer.add_scalar("voicepack/prosodic_norm", _prosodic_norm, epoch)
            log_info(
                f"acoustic_norm={_acoustic_norm:.4f} prosodic_norm={_prosodic_norm:.4f}"
            )
            if _vp is not None and _test_tokens:
                _epoch_audio = run_kokoro_inference(
                    model, _test_tokens, _vp, device, text_cleaner
                )
                for i, (text, audio) in enumerate(_epoch_audio):
                    writer.add_audio(
                        f"inference/test_{i + 1:02d}",
                        audio,
                        epoch,
                        sample_rate=sr,
                    )
            _ = [model[key].train() for key in model]
            # ───────────────────────────────────────────────────────────────────

    if accelerator.is_main_process:
        log_info("Saving..")
        state = {
            "net": {key: model[key].state_dict() for key in model},
            "optimizer": optimizer.state_dict(),
            "iters": iters,
            "val_loss": loss_test / iters_test,
            "epoch": epoch,
        }
        save_path = osp.join(log_dir, config.get("final_first_stage_path", "first_stage.pth"))
        atomic_torch_save(state, save_path)


if __name__ == "__main__":
    main()
