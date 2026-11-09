# load packages
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Keep PyTorch/MPS from expanding into most unified memory.  When WavLM/SLM
# adversarial training is active, uncapped MPS allocations can push macOS into
# heavy swap while process RSS still looks small.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.90")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.80")

REPO_ROOT = Path(__file__).resolve().parent
STYLETTS2_ROOT = REPO_ROOT / 'StyleTTS2'
if str(STYLETTS2_ROOT) not in sys.path:
    sys.path.insert(0, str(STYLETTS2_ROOT))

import copy
import gc
import random
import yaml
import time
from tqdm import tqdm
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
import click
import glob
import re
import shutil
import traceback
import warnings

warnings.simplefilter("ignore")
from torch.utils.tensorboard import SummaryWriter

from meldataset import build_dataloader
from kokoro_symbols import TextCleaner
from symbols_config import apply_symbols_config
from kokoro_tb_utils import prepare_test_tokens, extract_voicepack, run_kokoro_inference
from speaker_losses import SpeakerClassifier, SpeakerCenters, speaker_separation_loss

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer


# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


import logging
from logging import StreamHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


def _file_log(level, message, *args):
    if args:
        message = message % args
    record = logger.makeRecord(logger.name, level, __file__, 0, message, (), None)
    for handler in logger.handlers:
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


def find_latest_second_stage_checkpoint(log_dir):
    checkpoints = []
    pattern = osp.join(log_dir, "epoch_2nd_*.pth")
    for path in glob.glob(pattern):
        match = re.search(r"epoch_2nd_(\d+)\.pth$", osp.basename(path))
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
    '''Use a fixed waveform length for MPD/MSD to avoid MPSGraph cache growth.'''
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


def _resolve_aux_device(name, configured, device):
    '''Fall back to the training device when a config pins another backend.

    The VCTK config pins aux modules to "mps"; on a CUDA or CPU machine that
    backend is unavailable, so use the training device instead.
    '''
    configured = str(configured)
    backend = configured.split(':')[0]
    if backend in ('mps', 'cuda') and backend != str(device).split(':')[0]:
        log_warning(f"{name}={configured} does not match training device {device}; using {device}")
        return device
    return configured


# Pretrained helpers that Stage 2 runs under no_grad and never steps. Keeping
# them in eval mode during training gives deterministic targets: the pitch
# extractor's BatchNorm would otherwise use batch statistics for F0_real, and
# aligner/text-encoder dropout would perturb the alignment and decoder input.
FROZEN_MODULES = ('text_aligner', 'pitch_extractor', 'text_encoder')
# The decoder still trains after joint_epoch, but stays in eval mode as in
# upstream Stage 2: in train mode ISTFTNet randomly smooths the F0 and energy
# curves (Modules/istftnet.py Decoder.forward), which would change the audio
# loss that trains the prosody predictor.
EVAL_MODE_MODULES = FROZEN_MODULES + ('decoder',)


def _set_train_mode(model):
    for key in model:
        if key in EVAL_MODE_MODULES:
            model[key].eval()
        else:
            model[key].train()


def _style_ref_input(ref_mels, crop_frames, random_start=True):
    '''Padding-free style/prosody encoder input: returns (mels, lengths).

    The collater zero-pads mels to the fixed MPS width. Padding is not
    silence in normalized log-mel space, and the style encoders average over
    time, so unmasked padding dilutes the style vector and mismatches
    inference, where styles come from unpadded mels. True lengths are
    recovered from the content: padded frames are exactly zero.

    crop_frames <= 0 keeps the whole clip and returns its lengths, which the
    encoders use to mask padding (StyleEncoder(x, lengths)). crop_frames > 0
    takes a fixed-width circular crop inside the true length; it holds no
    padding, so lengths is None.
    '''
    lengths = (ref_mels.abs().sum(dim=1) > 0).sum(dim=-1).clamp(min=1)
    if not crop_frames or crop_frames <= 0:
        return ref_mels, lengths
    crop_frames = int(crop_frames)
    if random_start:
        starts = (torch.rand(ref_mels.size(0), device=ref_mels.device) * lengths).long()
    else:
        starts = torch.zeros_like(lengths)
    base = torch.arange(crop_frames, device=ref_mels.device).unsqueeze(0)
    idx = (starts.unsqueeze(1) + base) % lengths.unsqueeze(1)
    idx = idx.unsqueeze(1).expand(ref_mels.size(0), ref_mels.size(1), crop_frames)
    return torch.gather(ref_mels, dim=-1, index=idx), None


def _gb(value):
    return float(value) / 1024 ** 3


def _empty_accelerator_cache(device):
    '''Collect garbage and release cached accelerator memory.

    MPS keeps torch.mps.empty_cache(): since PyTorch 2.13 (PR #181485) it also
    clears the MPSGraph cache, and that has not been confirmed for the generic
    torch.accelerator.empty_cache().
    '''
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    elif device != 'cpu':
        torch.accelerator.empty_cache()


def _torch_memory_report_line(device):
    if device == 'mps' and torch.backends.mps.is_available():
        try:
            current = torch.mps.current_allocated_memory()
            driver = torch.mps.driver_allocated_memory()
            recommended = torch.mps.recommended_max_memory()
            return 'MPS: cur=%.2f drv=%.2f max=%.2f' % (_gb(current), _gb(driver), _gb(recommended))
        except Exception as exc:
            return f'MPS mem unavailable: {type(exc).__name__}'
    if device != 'cpu' and torch.accelerator.is_available():
        try:
            return '%s alloc=%.2f reserved=%.2f max=%.2f' % (
                device.upper(),
                _gb(torch.accelerator.memory_allocated()),
                _gb(torch.accelerator.memory_reserved()),
                _gb(torch.accelerator.max_memory_allocated()),
            )
        except Exception as exc:
            return f'{device.upper()} mem unavailable: {type(exc).__name__}'
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


def _duration_losses(d, d_gt, input_lengths):
    '''Fixed-shape duration L1 and CE losses.

    Equivalent to the original per-sample loop, but avoids variable-length
    slices on the training device: each distinct text length compiles a
    distinct MPS forward/backward graph that accumulates in driver memory.
    '''
    batch, text_width, max_dur = d.shape
    lengths = input_lengths.to(d.device)
    dur_tgt = d_gt.to(d.device).to(d.dtype)

    pos = torch.arange(text_width, device=d.device).unsqueeze(0)
    valid = (pos < lengths.unsqueeze(1)).to(d.dtype)
    inner = ((pos >= 1) & (pos < (lengths - 1).unsqueeze(1))).to(d.dtype)

    dur_pred = torch.sigmoid(d).sum(dim=-1)
    loss_dur = ((dur_pred - dur_tgt).abs() * inner).sum(dim=1) / inner.sum(dim=1).clamp(min=1)

    frame = torch.arange(max_dur, device=d.device).view(1, 1, max_dur)
    s2s_trg = (frame < dur_tgt.unsqueeze(-1)).to(d.dtype)
    ce = F.binary_cross_entropy_with_logits(d, s2s_trg, reduction='none').mean(dim=-1)
    loss_ce = (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    return loss_dur.sum() / batch, loss_ce.sum() / batch


@click.command()
@click.option("-p", "--config_path", default="configs/config_vctk_ft.yml", type=str)
@click.option('--dataset', type=str, default=None, help='Processed dataset directory from prepare_dataset.py')
@click.option('-name', '-n', '--name', type=str, default=None, help='Run/output name; saves checkpoints and logs under runs/<name>')
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from the latest epoch_2nd_*.pth checkpoint in log_dir.",
)
def main(config_path, dataset, name, resume):
    config_path = str((REPO_ROOT / config_path).resolve()) if not osp.isabs(config_path) else config_path
    config = yaml.safe_load(open(config_path))
    _resolve_runtime_paths(config, config_path, dataset, name)
    symbols_config = apply_symbols_config(config, config_path)

    log_dir = config["log_dir"]
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # Stage-specific name so Stage 1's copy of the config in the same run
    # directory is kept.
    config_stem, config_ext = osp.splitext(osp.basename(config_path))
    shutil.copy(config_path, osp.join(log_dir, f"{config_stem}_2nd{config_ext}"))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, "train.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s")
    )
    logger.addHandler(file_handler)
    if symbols_config:
        log_info(f'Using symbols: {symbols_config[0]} ({symbols_config[1]} tokens)')

    batch_size = config.get("batch_size", 10)

    epochs = config.get("epochs_2nd", 200)
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

    loss_params = Munch(config["loss_params"])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    slm_enabled = loss_params.lambda_slm > 0
    slmadv_enabled = slm_enabled and loss_params.get('slmadv_enabled', True)

    optimizer_params = Munch(config["optimizer_params"])

    train_list, val_list = get_data_path_list(train_path, val_path)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    log_info(f"Using {device} device")
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(int(config.get('mps_cpu_threads', 1)))
        log_info(f"Set torch CPU threads: {prev_threads} -> {torch.get_num_threads()} (mps_cpu_threads)")
        if bool(config.get('mps_native_lstm', False)):
            set_mps_native_lstm(True)
            log_info('Using native MPS packed LSTMs (mps_native_lstm=true)')
    log_info(f"Using device: {device} (MPS Fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')})")
    eval_device = _resolve_aux_device('eval_device', config.get("eval_device", "cpu" if device == "mps" else device), device)
    if eval_device != device:
        log_info(f"Using device: {eval_device} for validation and TensorBoard inference")
    elif device == 'mps':
        log_warning('Validation on MPS can reintroduce variable crop MPSGraph cache growth; prefer eval_device=cpu')
    aligner_device = _resolve_aux_device('aligner_device', config.get("aligner_device", "cpu" if device == "mps" else device), device)
    if aligner_device != device:
        log_info(f"Using device: {aligner_device} for text aligner to avoid variable-mel MPSGraph cache growth")
    discriminator_device = _resolve_aux_device('discriminator_device', config.get("discriminator_device", device), device)
    mpd_device = _resolve_aux_device('mpd_device', config.get("mpd_device", discriminator_device), device)
    msd_device = _resolve_aux_device('msd_device', config.get("msd_device", discriminator_device), device)
    if mpd_device != device:
        log_info(f"Using device: {mpd_device} for MPD discriminator")
    if msd_device != device:
        log_info(f"Using device: {msd_device} for MSD discriminator")
    wavlm_device = _resolve_aux_device('wavlm_device', config.get("wavlm_device", "cpu" if device == "mps" else device), device)
    if slm_enabled and wavlm_device != device:
        log_info(f"Using device: {wavlm_device} for frozen WavLM perceptual loss")
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
        log_info(f"Using fixed Stage 2 training mel length: {fixed_training_mel_len_frames} frames")
    fixed_mps_text_length = int(config.get('fixed_mps_text_length', 0) or 0)
    aligner_bucket_frames = int(config.get('aligner_bucket_frames', 64))
    fixed_mps_mel_length = int(config.get('fixed_mps_mel_length', 0) or 0)
    mps_safe_gather_crop = bool(config.get('mps_safe_gather_crop', device == 'mps'))
    style_ref_crop_frames = int(config.get('style_ref_crop_frames', 0) or 0)
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

    # load pretrained ASR model
    ASR_config = config.get("ASR_config", False)
    ASR_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(ASR_path, ASR_config, vocab_size=config['model_params']['n_token'])

    # load pretrained F0 model
    F0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(F0_path)

    # load PL-BERT model
    BERT_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(BERT_path, vocab_size=config['model_params']['n_token'])

    # build model
    model_params = recursive_munch(config["model_params"])
    multispeaker = model_params.multispeaker
    diffusion_enabled = loss_params.get('lambda_diff', 0) > 0 and diff_epoch < epochs
    model = build_model(model_params, text_aligner, pitch_extractor, plbert, enable_diffusion=diffusion_enabled)
    if not diffusion_enabled:
        log_info('Diffusion model disabled; skipping diffusion module allocation')
    if not slm_enabled:
        log_info('SLM/WavLM loss and adversarial updates disabled (lambda_slm <= 0)')
    elif not slmadv_enabled:
        log_info('WavLM loss enabled; SLM adversarial updates disabled (slmadv_enabled=false)')
    _ = [model[key].to(device) for key in model]
    if aligner_device != device:
        model.text_aligner.to(aligner_device)

    start_epoch = 0
    iters = 0

    resume_checkpoint = None
    if resume:
        latest_checkpoint = find_latest_second_stage_checkpoint(log_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(
                f"--resume was set, but no epoch_2nd_*.pth checkpoints were found in {log_dir}"
            )
        resume_epoch, resume_checkpoint = latest_checkpoint
        log_info(
            f"Resuming from latest second-stage checkpoint: {resume_checkpoint} "
            f"(epoch {resume_epoch})"
        )

    load_pretrained = resume_checkpoint is not None or (
        config.get("pretrained_model", "") != "" and config.get(
            "second_stage_load_pretrained", False
        )
    )

    if not load_pretrained:
        if config.get("first_stage_path", "") != "":
            first_stage_path = osp.join(
                log_dir, config.get("first_stage_path", "first_stage.pth")
            )
            log_info("Loading the first stage model at %s ..." % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(
                model,
                None,
                first_stage_path,
                load_only_params=True,
                ignore_modules=[
                    "predictor_encoder",
                    "msd",
                    "mpd",
                    "wd",
                    "diffusion",
                ],
            )  # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch

            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError("You need to specify the path to the first stage model.")

    num_speakers = int(config.get('num_speakers') or _num_speakers_from_lists(train_list, val_list))
    style_dim = int(config['model_params'].get('style_dim', 128))
    if loss_params.get('lambda_prosody_spk', 0) > 0:
        model['prosody_speaker_classifier'] = SpeakerClassifier(style_dim, num_speakers)
    if loss_params.get('lambda_prosody_center', 0) > 0 or loss_params.get('lambda_speaker_margin', 0) > 0:
        model['prosody_speaker_centers'] = SpeakerCenters(num_speakers, style_dim)
    if 'prosody_speaker_classifier' in model:
        model['prosody_speaker_classifier'].to(device)
    if 'prosody_speaker_centers' in model:
        model['prosody_speaker_centers'].to(device)
    if 'prosody_speaker_classifier' in model or 'prosody_speaker_centers' in model:
        log_info(f'Enabled prosody speaker separation loss for {num_speakers} speakers')

    if mpd_device != device:
        model.mpd.to(mpd_device)
    if msd_device != device:
        model.msd.to(msd_device)
    gl = GeneratorLoss(model.mpd, model.msd)
    dl = DiscriminatorLoss(model.mpd, model.msd)
    wl = WavLMLoss(model_params.slm.model, model.wd, sr, model_params.slm.sr).to(wavlm_device) if slm_enabled else None

    if device == "cuda":
        gl = MyDataParallel(gl)
        dl = MyDataParallel(dl)
        if slm_enabled:
            wl = MyDataParallel(wl)

    def _restore_training_devices():
        '''Move every module back to its configured training device after eval.'''
        _ = [model[key].to(device) for key in model]
        if aligner_device != device:
            model.text_aligner.to(aligner_device)
        if mpd_device != device:
            model.mpd.to(mpd_device)
        if msd_device != device:
            model.msd.to(msd_device)
        if wl is not None:
            # wl.wd is model.wd, so the loop above moved it to `device`.
            wl.to(wavlm_device)

    sampler = None
    if diffusion_enabled:
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001, sigma_max=3.0, rho=9.0
            ),  # empirical parameters
            clamp=False,
        )

    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict["bert"]["max_lr"] = optimizer_params.bert_lr * 2
    scheduler_params_dict["decoder"]["max_lr"] = optimizer_params.ft_lr * 2
    scheduler_params_dict["style_encoder"]["max_lr"] = optimizer_params.ft_lr * 2
    if "prosody_speaker_classifier" in scheduler_params_dict:
        scheduler_params_dict["prosody_speaker_classifier"]["max_lr"] = optimizer_params.ft_lr * 2
    if "prosody_speaker_centers" in scheduler_params_dict:
        scheduler_params_dict["prosody_speaker_centers"]["max_lr"] = optimizer_params.ft_lr * 2

    optimizer = build_optimizer(
        {key: model[key].parameters() for key in model},
        scheduler_params_dict=scheduler_params_dict,
        lr=optimizer_params.lr,
    )

    # adjust BERT learning rate
    for g in optimizer.optimizers["bert"].param_groups:
        g["betas"] = (0.9, 0.99)
        g["lr"] = optimizer_params.bert_lr
        g["initial_lr"] = optimizer_params.bert_lr
        g["min_lr"] = 0
        g["weight_decay"] = 0.01

    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g["betas"] = (0.0, 0.99)
            g["lr"] = optimizer_params.ft_lr
            g["initial_lr"] = optimizer_params.ft_lr
            g["min_lr"] = 0
            g["weight_decay"] = 1e-4

    # load models if there is a model
    if load_pretrained:
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
            start_epoch += 1
            log_info(f"Resuming second-stage training from epoch {start_epoch}/{epochs - 1}")
        # Ensure all modules are in train() mode after loading
        # (load_checkpoint sets them to eval(), which breaks spectral_norm)
        _set_train_mode(model)

    # X23/X24: keep the decoder's waveform generator at the loaded weights (Stage 1 with freeze_generator leaves it
    # at base). Gradients still flow through it; its parameters get no grad, so AdamW (incl. weight decay) skips them.
    if config.get("freeze_generator", False):
        model.decoder.generator.requires_grad_(False)
        log_info("Decoder generator frozen for the whole run (freeze_generator)")

    # DP — must happen AFTER load_checkpoint so state dict keys match
    # (DataParallel adds 'module.' prefix which breaks strict=False loading)
    if device == "cuda":
        for key in model:
            if key != "mpd" and key != "msd" and key != "wd":
                model[key] = MyDataParallel(model[key])

    n_down = model.text_aligner.n_down

    best_loss = float("inf")  # best validation loss seen so far
    if resume_checkpoint is not None:
        best_path = osp.join(log_dir, config.get("best_second_stage_path", "best_2nd.pth"))
        best_source = best_path if osp.exists(best_path) else resume_checkpoint
        try:
            best_value = torch.load(best_source, map_location="cpu").get("val_loss", best_loss)
            best_loss = float(best_value.item() if torch.is_tensor(best_value) else best_value)
            log_info("Loaded best Stage 2 validation loss %.6f from %s", best_loss, best_source)
        except Exception:
            log_warning("Could not load best Stage 2 validation loss from %s", best_source)
    loss_train_record = list([])
    loss_test_record = list([])
    iters = 0

    criterion = nn.L1Loss()  # F0 loss (regression)
    _empty_accelerator_cache(device)

    stft_loss = MultiResolutionSTFTLoss().to(device)

    log_info("BERT %s", optimizer.optimizers["bert"])
    log_info("decoder %s", optimizer.optimizers["decoder"])

    start_ds = False

    running_std = []

    slmadv_params = Munch(config["slmadv_params"])
    slmadv = None
    # On MPS the slmadv step leaks native driver memory (~7 MB per invocation;
    # WavLM grad-forward window, not reproducible standalone). Running it every
    # N steps bounds the leak to fit a full run under the MPS watermark while
    # keeping most of the adversarial signal.
    slmadv_every = max(int(slmadv_params.get('every', 1) or 1), 1)
    slmadv_calls = 0
    if slmadv_enabled:
        slmadv = SLMAdversarialLoss(
            model,
            wl,
            sampler,
            slmadv_params.min_len,
            slmadv_params.max_len,
            batch_percentage=slmadv_params.batch_percentage,
            skip_update=slmadv_params.iter,
            sig=slmadv_params.sig,
            diffusion_enabled=diffusion_enabled,
            fixed_mel_len=slmadv_params.get('slm_fixed_mel_len', None),
            fixed_crop_size=slmadv_params.get('slm_fixed_crop_size', None),
            conv_on_cpu=slmadv_params.get('slm_conv_on_cpu', False),
        )
        log_info(
            'SLM adversarial enabled: fixed_mel_len=%s fixed_crop_size=%s conv_on_cpu=%s every=%d'
            % (
                slmadv_params.get('slm_fixed_mel_len', None),
                slmadv_params.get('slm_fixed_crop_size', None),
                slmadv_params.get('slm_conv_on_cpu', False),
                slmadv_every,
            )
        )

    # ── Kokoro-faithful TensorBoard inference setup ───────────────────────────
    text_cleaner = TextCleaner()
    _test_tokens = prepare_test_tokens(text_cleaner, OOD_data)
    if _test_tokens:
        log_info(
            f"TensorBoard inference: prepared {len(_test_tokens)} test sentences"
        )
    else:
        log_warning(
            "TensorBoard inference: no test sentences available (misaki/espeak not found?)"
        )

    # Generate Stage 1 baseline before any training updates
    if eval_device != device:
        _ = [model[key].to(eval_device) for key in model]
    _ = [model[key].eval() for key in model]
    log_info("Extracting Stage 1 baseline voicepack for TensorBoard...")
    _baseline_voicepack, _baseline_acoustic_norm, _baseline_prosodic_norm = (
        extract_voicepack(
            model,
            root_path,
            eval_device,
            n_samples=20,
        )
    )
    if _baseline_voicepack is not None:
        writer.add_scalar("baseline/acoustic_norm", _baseline_acoustic_norm, 0)
        writer.add_scalar("baseline/prosodic_norm", _baseline_prosodic_norm, 0)
        log_info(
            f"  acoustic_norm={_baseline_acoustic_norm:.4f}  prosodic_norm={_baseline_prosodic_norm:.4f}"
        )
        _baseline_audio = run_kokoro_inference(
            model, _test_tokens, _baseline_voicepack, eval_device, text_cleaner
        )
        for i, (text, audio) in enumerate(_baseline_audio):
            writer.add_audio(f"baseline/test_{i + 1:02d}", audio, 0, sample_rate=sr)
            log_info(f"  baseline/test_{i + 1:02d}: {text[:60]}")
    if eval_device != device:
        _restore_training_devices()
    _set_train_mode(model)
    # ─────────────────────────────────────────────────────────────────────────

    epoch_durations = []
    warned_mps_mel_width = False
    for epoch in range(start_epoch, epochs):
        if device == "mps":
            _empty_accelerator_cache(device)
        running_loss = 0
        running_count = 0
        skipped_short_batches = 0
        dropped_short_samples = 0
        skipped_aligner_batches = 0
        start_time = time.time()

        _ = [model[key].eval() for key in model]

        _set_train_mode(model)

        if epoch >= joint_epoch:
            start_ds = True

        train_progress = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc='Epoch %d/%d' % (epoch, epochs - 1),
            unit='step',
            dynamic_ncols=True,
        )
        for i, batch in train_progress:
            waves = batch[0]
            labels = batch[-2].to(device)
            ref_labels = batch[-1].to(device)
            batch_cpu = batch[1:-2]
            aligner_texts = batch_cpu[0].to(aligner_device)
            aligner_mels = batch_cpu[4].to(aligner_device)
            aligner_mel_input_length = batch_cpu[5].to(aligner_device)
            batch = [b.to(device) for b in batch_cpu]
            (
                texts,
                input_lengths,
                ref_texts,
                ref_lengths,
                mels,
                mel_input_length,
                ref_mels,
            ) = batch

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
                    keep_idx_al = keep_idx.to(aligner_device)
                    aligner_texts = aligner_texts[keep_idx_al]
                    aligner_mels = aligner_mels[keep_idx_al]
                    aligner_mel_input_length = aligner_mel_input_length[keep_idx_al]
                    texts = texts[keep_idx]
                    input_lengths = input_lengths[keep_idx]
                    ref_texts = ref_texts[keep_idx]
                    ref_lengths = ref_lengths[keep_idx]
                    mels = mels[keep_idx]
                    mel_input_length = mel_input_length[keep_idx]
                    ref_mels = ref_mels[keep_idx]

            with torch.no_grad():
                text_mask = _length_to_text_mask(input_lengths, texts.size(1)).to(texts.device)

                try:
                    _, s2s_attn = run_text_aligner(
                        model.text_aligner,
                        aligner_mels.detach(),
                        aligner_mel_input_length,
                        aligner_texts,
                        n_down,
                        aligner_bucket_frames,
                    )
                    s2s_attn = s2s_attn.transpose(-1, -2)
                    s2s_attn = s2s_attn[..., 1:]
                    s2s_attn = s2s_attn.transpose(-1, -2).to(device)
                except Exception as exc:
                    skipped_aligner_batches += 1
                    if skipped_aligner_batches <= 3:
                        log_warning(
                            'Skipping batch after text_aligner error: %s: %s; text_min=%d text_max=%d emb=%d\n%s',
                            type(exc).__name__,
                            exc,
                            int(aligner_texts.min().item()),
                            int(aligner_texts.max().item()),
                            int(model.text_aligner.asr_s2s.embedding.num_embeddings),
                            traceback.format_exc(),
                        )
                    continue

                mask_ST = mask_from_lens(
                    s2s_attn, input_lengths, mel_input_length // (2**n_down)
                )
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                # encode
                t_en = model.text_encoder(texts, input_lengths, text_mask)
                if t_en is not None and torch.isnan(t_en).any():
                    log_error(
                        "NaN detected in t_en after t_en = model.text_encoder(texts, input_lengths, text_mask)"
                    )
                    import sys

                    sys.exit(1)
                asr = t_en @ s2s_attn_mono
                if asr is not None and torch.isnan(asr).any():
                    log_error("NaN detected in asr after asr = t_en @ s2s_attn_mono")
                    import sys

                    sys.exit(1)

                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                # Padding-free, fixed-shape style-encoder input (see _style_ref_input).
                ref_style_mels, ref_style_lengths = _style_ref_input(ref_mels, style_ref_crop_frames)

                ref = None

                # compute reference styles
                if multispeaker and epoch >= diff_epoch:
                    ref_ss = model.style_encoder(ref_style_mels.unsqueeze(1), ref_style_lengths)
                    if ref_ss is not None and torch.isnan(ref_ss).any():
                        log_error("NaN detected in ref_ss after ref_ss = model.style_encoder(")
                        import sys

                        sys.exit(1)
                    ref_sp = model.predictor_encoder(ref_style_mels.unsqueeze(1), ref_style_lengths)
                    ref = torch.cat([ref_ss, ref_sp], dim=1)

            # Target styles for the denoiser/SLM adversarial path. Outputs are
            # only consumed detached, so run gradient-free; the circular crop
            # keeps shapes fixed and avoids both collation padding and the old
            # per-sample raw-length MPS graph variants.
            need_s_trg = diffusion_enabled or slmadv_enabled
            if need_s_trg:
                with torch.no_grad():
                    gt_style_mels, gt_style_lengths = _style_ref_input(mels, style_ref_crop_frames)
                    s_trg_dur = model.predictor_encoder(gt_style_mels.unsqueeze(1), gt_style_lengths)  # global prosodic styles
                    s_trg_acoustic = model.style_encoder(gt_style_mels.unsqueeze(1), gt_style_lengths)  # global acoustic styles
                    s_trg = torch.cat([s_trg_acoustic, s_trg_dur], dim=-1).detach()  # ground truth for denoiser
            else:
                s_trg = None

            bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
            if bert_dur is not None and torch.isnan(bert_dur).any():
                log_error(
                    "NaN detected in bert_dur after bert_dur = model.bert(texts, attention_mask=(~text_mask).int())"
                )
                import sys

                sys.exit(1)
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            if d_en is not None and torch.isnan(d_en).any():
                log_error(
                    "NaN detected in d_en after d_en = model.bert_encoder(bert_dur).transpose(-1, -2)"
                )
                import sys

                sys.exit(1)

            # As in upstream StyleTTS2, duration is conditioned on the whole
            # target utterance. Unlike upstream, which uses a crop of that same
            # target, F0/energy (F0Ntrain below) and the decoder are conditioned
            # on ref_mels, a different recording of the same speaker in
            # multispeaker runs. Length masking keeps the target input
            # fixed-shape and excludes the collation padding.
            s_dur_target = model.predictor_encoder(mels.detach().unsqueeze(1), mel_input_length)
            if multispeaker:
                s_dur = model.predictor_encoder(ref_style_mels.unsqueeze(1), ref_style_lengths)
            else:
                own_style_mels, own_style_lengths = _style_ref_input(mels.detach(), style_ref_crop_frames)
                s_dur = model.predictor_encoder(own_style_mels.unsqueeze(1), own_style_lengths)

            # denoiser training
            if diffusion_enabled and epoch >= diff_epoch:
                num_steps = np.random.randint(3, 5)

                if model_params.diffusion.dist.estimate_sigma_data:
                    model.diffusion.diffusion.sigma_data = (
                        s_trg.std(axis=-1).mean().item()
                    )  # batch-wise std estimation
                    running_std.append(model.diffusion.diffusion.sigma_data)

                if multispeaker:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        features=ref,  # reference from the same speaker as the embedding
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur, features=ref
                    ).mean()  # EDM loss
                    loss_sty = F.l1_loss(
                        s_preds, s_trg.detach()
                    )  # style reconstruction loss
                else:
                    s_preds = sampler(
                        noise=torch.randn_like(s_trg).unsqueeze(1).to(device),
                        embedding=bert_dur,
                        embedding_scale=1,
                        embedding_mask_proba=0.1,
                        num_steps=num_steps,
                    ).squeeze(1)
                    loss_diff = model.diffusion(
                        s_trg.unsqueeze(1), embedding=bert_dur
                    ).mean()  # EDM loss
                    loss_sty = F.l1_loss(
                        s_preds, s_trg.detach()
                    )  # style reconstruction loss
            else:
                loss_sty = 0
                loss_diff = 0

            d, p = model.predictor(d_en, s_dur_target, input_lengths, s2s_attn_mono, text_mask)
            if p is not None and torch.isnan(p).any():
                log_error(
                    "NaN detected in p after d, p = model.predictor(d_en, s_dur_target, input_lengths, s2s_attn_mono, text_mask)"
                )
                import sys

                sys.exit(1)
            if d is not None and torch.isnan(d).any():
                log_error(
                    "NaN detected in d after d, p = model.predictor(d_en, s_dur_target, input_lengths, s2s_attn_mono, text_mask)"
                )
                import sys

                sys.exit(1)

            if fixed_training_mel_len:
                # Pin decoder/STFT/WavLM shapes on MPS. Variable mel_len creates
                # distinct MPSGraph executables that accumulate in driver/native memory.
                # Too-short samples were already dropped at the top of the loop.
                mel_len = fixed_training_mel_len_frames
            else:
                mel_len = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)
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
                source_mel_width = mels.size(-1)
                source_asr_width = asr.size(-1)
                if fixed_mps_mel_length > 0 and source_mel_width > fixed_mps_mel_length and not warned_mps_mel_width:
                    log_warning(
                        'Batch mel width %d exceeds fixed_mps_mel_length=%d; consider increasing the fixed width to avoid extra MPS graph variants',
                        source_mel_width,
                        fixed_mps_mel_length,
                    )
                    warned_mps_mel_width = True
                en = _time_gather(_pad_last(asr, source_asr_width), starts_t, mel_len)
                p_en = _time_gather(_pad_last(p, source_asr_width), starts_t, mel_len)
                gt = _time_gather(_pad_last(mels, source_mel_width), starts_t * 2, mel_len * 2).detach()
            else:
                en = torch.stack([asr[bib, :, s0 : s0 + mel_len] for bib, s0 in enumerate(starts)])
                p_en = torch.stack([p[bib, :, s0 : s0 + mel_len] for bib, s0 in enumerate(starts)])
                gt = torch.stack([
                    mels[bib, :, (s0 * 2) : ((s0 + mel_len) * 2)] for bib, s0 in enumerate(starts)
                ]).detach()
            st = ref_style_mels  # Use actual reference from different clip

            if gt.size(-1) < 80:
                continue

            with torch.no_grad():
                F0_real, _, _ = model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)

            prosody_spk_loss, prosody_spk_ce, prosody_center_loss, prosody_margin_loss = speaker_separation_loss(
                s_dur,
                ref_labels if multispeaker else labels,
                classifier=model.get('prosody_speaker_classifier'),
                centers=model.get('prosody_speaker_centers'),
                lambda_ce=loss_params.get('lambda_prosody_spk', 0),
                lambda_center=loss_params.get('lambda_prosody_center', 0),
                lambda_margin=loss_params.get('lambda_speaker_margin', 0),
                margin=loss_params.get('speaker_center_margin', 1.0),
            )
            s = (
                model.style_encoder(st.unsqueeze(1), ref_style_lengths)
                if multispeaker
                else model.style_encoder(gt.unsqueeze(1))
            )

            y_rec_gt = wav.unsqueeze(1)
            y_rec_gt_pred = None

            if epoch >= joint_epoch:
                # ground truth from recording
                wav = y_rec_gt  # use recording since decoder is tuned
            else:
                # Ground truth from reconstruction since decoder is fixed.
                # This is a target branch; do not keep a second decoder autograd graph.
                with torch.no_grad():
                    y_rec_gt_pred = model.decoder(en, F0_real, N_real, s).detach()
                wav = y_rec_gt_pred

            F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

            y_rec = model.decoder(en, F0_fake, N_fake, s)

            loss_F0_rec = (F.smooth_l1_loss(F0_real, F0_fake)) / 10
            loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)

            if start_ds:
                wav_adv, y_rec_adv = _crop_adversarial_audio(wav, y_rec, adversarial_segment_size)
                wav_adv_disc = wav_adv.detach()
                y_rec_adv_disc_detached = y_rec_adv.detach()
                optimizer.zero_grad()
                d_loss = dl(wav_adv_disc, y_rec_adv_disc_detached).mean()
                d_loss.backward()
                optimizer.step("msd")
                optimizer.step("mpd")
                del wav_adv_disc, y_rec_adv_disc_detached
            else:
                d_loss = 0

            # generator loss
            optimizer.zero_grad()
            discriminator_params_frozen = False

            loss_mel = stft_loss(y_rec, wav)
            if loss_mel is not None and torch.isnan(loss_mel).any():
                log_error("NaN detected in loss_mel after loss_mel = stft_loss(y_rec, wav)")
                import sys

                sys.exit(1)
            if start_ds:
                # The generator adversarial loss must backprop through the
                # discriminator stack to y_rec_adv, but discriminator parameter
                # gradients are unused in this phase.  Disabling them avoids
                # stale grad buffers, especially with split MPD/MSD devices.
                for p_disc in list(model.mpd.parameters()) + list(model.msd.parameters()):
                    p_disc.requires_grad_(False)
                discriminator_params_frozen = True
                loss_gen_all = gl(
                    wav_adv.detach(),
                    y_rec_adv,
                ).mean().to(device)
            else:
                loss_gen_all = 0
            if slm_enabled:
                wav_lm, y_rec_lm = _crop_adversarial_audio(wav.detach(), y_rec, slm_segment_size)
                loss_lm = wl(
                    wav_lm.squeeze(1).to(wavlm_device),
                    y_rec_lm.squeeze(1).to(wavlm_device),
                ).mean().to(device)
            else:
                loss_lm = 0

            loss_dur, loss_ce = _duration_losses(d, d_gt, input_lengths)

            g_loss = (
                loss_params.lambda_mel * loss_mel
                + loss_params.lambda_F0 * loss_F0_rec
                + loss_params.lambda_ce * loss_ce
                + loss_params.lambda_norm * loss_norm_rec
                + loss_params.lambda_dur * loss_dur
                + loss_params.lambda_gen * loss_gen_all
                + loss_params.lambda_slm * loss_lm
                + loss_params.lambda_sty * loss_sty
                + loss_params.lambda_diff * loss_diff
                + prosody_spk_loss
            )

            running_loss += loss_mel.item()
            running_count += 1
            g_loss.backward()
            if discriminator_params_frozen:
                for p_disc in list(model.mpd.parameters()) + list(model.msd.parameters()):
                    p_disc.requires_grad_(True)

            optimizer.step("bert_encoder")
            optimizer.step("bert")
            optimizer.step("predictor")
            optimizer.step("predictor_encoder")
            if 'prosody_speaker_classifier' in model:
                optimizer.step("prosody_speaker_classifier")
            if 'prosody_speaker_centers' in model:
                optimizer.step("prosody_speaker_centers")

            if diffusion_enabled and epoch >= diff_epoch:
                optimizer.step("diffusion")

            if epoch >= joint_epoch:
                optimizer.step("style_encoder")
                optimizer.step("decoder")

            if epoch >= joint_epoch and slmadv_enabled and i % slmadv_every == 0:
                # randomly pick whether to use in-distribution text
                if np.random.rand() < 0.5:
                    use_ind = True
                else:
                    use_ind = False

                if use_ind:
                    ref_lengths = input_lengths
                    ref_texts = texts

                if y_rec_gt_pred is None:
                    with torch.no_grad():
                        y_rec_gt_pred = model.decoder(en, F0_real, N_real, s).detach()
                # Count adversarial calls, not loader steps: with every=N the
                # discriminator's own skip_update schedule would otherwise never
                # line up with the steps that reach this block.
                slmadv_calls += 1
                slm_out = slmadv(
                    slmadv_calls,
                    y_rec_gt,
                    y_rec_gt_pred,
                    waves,
                    mel_input_length,
                    ref_texts,
                    ref_lengths,
                    use_ind,
                    s_trg.detach(),
                    ref,
                )

                # None: too few clips long enough for the SLM crop. Skip only
                # the SLM update; the main step already ran, so still count it.
                if slm_out is None:
                    d_loss_slm, loss_gen_lm = 0, 0
                else:
                    d_loss_slm, loss_gen_lm, y_pred = slm_out

                    # SLM generator loss
                    optimizer.zero_grad()
                    loss_gen_lm.backward()

                    # compute the gradient norm
                    total_norm = {}
                    for key in model.keys():
                        total_norm[key] = 0
                        parameters = [
                            p
                            for p in model[key].parameters()
                            if p.grad is not None and p.requires_grad
                        ]
                        for p in parameters:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm[key] += param_norm.item() ** 2
                        total_norm[key] = total_norm[key] ** 0.5

                    # gradient scaling
                    if total_norm["predictor"] > slmadv_params.thresh:
                        for key in model.keys():
                            for p in model[key].parameters():
                                if p.grad is not None:
                                    p.grad *= 1 / total_norm["predictor"]

                    for p in model.predictor.duration_proj.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    for p in model.predictor.lstm.parameters():
                        if p.grad is not None:
                            p.grad *= slmadv_params.scale

                    if diffusion_enabled:
                        for p in model.diffusion.parameters():
                            if p.grad is not None:
                                p.grad *= slmadv_params.scale

                    optimizer.step("bert_encoder")
                    optimizer.step("bert")
                    optimizer.step("predictor")
                    if diffusion_enabled:
                        optimizer.step("diffusion")

                    # SLM discriminator loss
                    if d_loss_slm != 0:
                        optimizer.zero_grad()
                        # d_loss_slm only trains the WavLM discriminator on detached
                        # audio/features, so retaining the generator graph here is
                        # unnecessary and can accumulate MPS unified memory.
                        d_loss_slm.backward()
                        optimizer.step("wd")

                    d_loss_slm_value = (
                        d_loss_slm.item() if torch.is_tensor(d_loss_slm) else d_loss_slm
                    )
                    loss_gen_lm_value = (
                        loss_gen_lm.item() if torch.is_tensor(loss_gen_lm) else loss_gen_lm
                    )
                    del slm_out, y_pred, loss_gen_lm, d_loss_slm
                    d_loss_slm, loss_gen_lm = d_loss_slm_value, loss_gen_lm_value

            else:
                d_loss_slm, loss_gen_lm = 0, 0

            mps_cache_interval = config.get("mps_empty_cache_interval", 25)
            if (
                device == "mps"
                and mps_cache_interval > 0
                and (i + 1) % mps_cache_interval == 0
            ):
                _empty_accelerator_cache(device)

            iters = iters + 1

            train_progress.set_postfix(
                {
                    'Loss': '%.2f' % scalar_value(loss_mel),
                    'Disc': '%.2f' % scalar_value(d_loss),
                    'Dur': '%.2f' % scalar_value(loss_dur),
                    'CE': '%.2f' % scalar_value(loss_ce),
                    'Norm': '%.2f' % scalar_value(loss_norm_rec),
                    'F0': '%.2f' % scalar_value(loss_F0_rec),
                    'LM': '%.2f' % scalar_value(loss_lm),
                    'Gen': '%.2f' % scalar_value(loss_gen_all),
                    'Sty': '%.2f' % scalar_value(loss_sty),
                    'Diff': '%.2f' % scalar_value(loss_diff),
                    'DiscLM': '%.2f' % scalar_value(d_loss_slm),
                    'GenLM': '%.2f' % scalar_value(loss_gen_lm),
                    'ProsodyCE': '%.2f' % scalar_value(prosody_spk_ce),
                    'ProsodyCenter': '%.2f' % scalar_value(prosody_center_loss),
                    'SpkMargin': '%.2f' % scalar_value(prosody_margin_loss),
                    'MPS': _torch_memory_report_line(device).replace('MPS: ', ''),
                },
                refresh=True,
            )

            if (i + 1) % log_interval == 0:
                # Skipped batches add no loss, so divide by the batches counted.
                avg_mel_loss = running_loss / max(running_count, 1)
                elapsed = _format_elapsed(time.time() - start_time)
                memory_line = _torch_memory_report_line(device)
                step_message = (
                    "Step %d/%d: Loss=%.2f Disc=%.2f Dur=%.2f CE=%.2f Norm=%.2f F0=%.2f LM=%.2f Gen=%.2f Sty=%.2f Diff=%.2f DiscLM=%.2f GenLM=%.2f ProsodyCE=%.2f ProsodyCenter=%.2f SpkMargin=%.2f Elapsed=%s\n%s"
                    % (
                        i + 1,
                        len(train_dataloader),
                        avg_mel_loss,
                        d_loss,
                        loss_dur,
                        loss_ce,
                        loss_norm_rec,
                        loss_F0_rec,
                        loss_lm,
                        loss_gen_all,
                        loss_sty,
                        loss_diff,
                        d_loss_slm,
                        loss_gen_lm,
                        prosody_spk_ce,
                        prosody_center_loss,
                        prosody_margin_loss,
                        elapsed,
                        memory_line,
                    )
                )
                _file_log(logging.INFO, step_message)
                train_progress.set_postfix(
                    {
                        'Loss': '%.2f' % avg_mel_loss,
                        'Disc': '%.2f' % scalar_value(d_loss),
                        'Dur': '%.2f' % scalar_value(loss_dur),
                        'CE': '%.2f' % scalar_value(loss_ce),
                        'Norm': '%.2f' % scalar_value(loss_norm_rec),
                        'F0': '%.2f' % scalar_value(loss_F0_rec),
                        'LM': '%.2f' % scalar_value(loss_lm),
                        'Gen': '%.2f' % scalar_value(loss_gen_all),
                        'Sty': '%.2f' % scalar_value(loss_sty),
                        'Diff': '%.2f' % scalar_value(loss_diff),
                        'DiscLM': '%.2f' % scalar_value(d_loss_slm),
                        'GenLM': '%.2f' % scalar_value(loss_gen_lm),
                        'ProsodyCE': '%.2f' % scalar_value(prosody_spk_ce),
                        'ProsodyCenter': '%.2f' % scalar_value(prosody_center_loss),
                        'SpkMargin': '%.2f' % scalar_value(prosody_margin_loss),
                        'MPS': memory_line.replace('MPS: ', ''),
                    },
                    refresh=True,
                )

                writer.add_scalar("train/mel_loss", avg_mel_loss, iters)
                writer.add_scalar("train/gen_loss", loss_gen_all, iters)
                writer.add_scalar("train/d_loss", d_loss, iters)
                writer.add_scalar("train/ce_loss", loss_ce, iters)
                writer.add_scalar("train/dur_loss", loss_dur, iters)
                writer.add_scalar("train/slm_loss", loss_lm, iters)
                writer.add_scalar("train/norm_loss", loss_norm_rec, iters)
                writer.add_scalar("train/F0_loss", loss_F0_rec, iters)
                writer.add_scalar("train/sty_loss", loss_sty, iters)
                writer.add_scalar("train/diff_loss", loss_diff, iters)
                writer.add_scalar("train/d_loss_slm", d_loss_slm, iters)
                writer.add_scalar("train/gen_loss_slm", loss_gen_lm, iters)
                writer.add_scalar("train/prosody_spk_ce", prosody_spk_ce, iters)
                writer.add_scalar("train/prosody_center_loss", prosody_center_loss, iters)
                writer.add_scalar("train/speaker_margin_loss", prosody_margin_loss, iters)

                running_loss = 0
                running_count = 0

        if skipped_short_batches or dropped_short_samples or skipped_aligner_batches:
            log_info(
                'Epoch=%d/%d skipped_short_batches=%d dropped_short_samples=%d skipped_aligner_batches=%d fixed_mel_len=%d'
                % (epoch, epochs - 1, skipped_short_batches, dropped_short_samples, skipped_aligner_batches, fixed_training_mel_len_frames)
            )

        loss_test = 0
        loss_align = 0
        loss_f = 0
        if eval_device != device:
            _ = [model[key].to(eval_device) for key in model]
            stft_loss = stft_loss.to(eval_device)
        _ = [model[key].eval() for key in model]
        # When the whole model is moved to eval_device, the text aligner moves
        # with it; keep feeding it there instead of the training-time
        # aligner_device (which may be MPS in all-MPS experiments).
        val_aligner_device = eval_device if eval_device != device else aligner_device

        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                try:
                    waves = batch[0]
                    batch = [b.to(eval_device) for b in batch[1:-2]]
                    (
                        texts,
                        input_lengths,
                        ref_texts,
                        ref_lengths,
                        mels,
                        mel_input_length,
                        ref_mels,
                    ) = batch
                    with torch.no_grad():
                        aligner_mels = mels.detach().to(val_aligner_device)
                        text_mask = _length_to_text_mask(input_lengths, texts.size(1)).to(texts.device)

                        _, s2s_attn = run_text_aligner(
                            model.text_aligner,
                            aligner_mels,
                            mel_input_length.to(val_aligner_device),
                            texts.to(val_aligner_device),
                            n_down,
                            aligner_bucket_frames,
                        )
                        s2s_attn = s2s_attn.transpose(-1, -2)
                        s2s_attn = s2s_attn[..., 1:]
                        s2s_attn = s2s_attn.transpose(-1, -2).to(eval_device)

                        mask_ST = mask_from_lens(
                            s2s_attn, input_lengths, mel_input_length // (2**n_down)
                        )
                        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)
                        asr = t_en @ s2s_attn_mono

                        d_gt = s2s_attn_mono.sum(axis=-1).detach()

                    # Same duration conditioning as training: the whole target
                    # utterance, with the collation padding masked.
                    s = model.predictor_encoder(mels.unsqueeze(1), mel_input_length)

                    bert_dur = model.bert(texts, attention_mask=(~text_mask).int())
                    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
                    d, p = model.predictor(
                        d_en, s, input_lengths, s2s_attn_mono, text_mask
                    )
                    # get clips
                    mel_len = int(mel_input_length.min().item() / 2 - 1)
                    en = []
                    gt = []
                    p_en = []
                    wav = []

                    for bib in range(len(mel_input_length)):
                        mel_length = int(mel_input_length[bib].item() / 2)

                        random_start = np.random.randint(0, mel_length - mel_len)
                        en.append(asr[bib, :, random_start : random_start + mel_len])
                        p_en.append(p[bib, :, random_start : random_start + mel_len])

                        gt.append(
                            mels[
                                bib,
                                :,
                                (random_start * 2) : ((random_start + mel_len) * 2),
                            ]
                        )

                        y = waves[bib][
                            (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
                        ]
                        wav.append(torch.from_numpy(y).float().to(eval_device))

                    wav = torch.stack(wav).float().detach()


                    en = torch.stack(en)
                    p_en = torch.stack(p_en)
                    gt = torch.stack(gt).detach()

                    val_ref_style_mels, val_ref_style_lengths = _style_ref_input(
                        ref_mels.detach(), style_ref_crop_frames, random_start=False
                    )
                    s_dur = (
                        model.predictor_encoder(val_ref_style_mels.unsqueeze(1), val_ref_style_lengths)
                        if multispeaker
                        else model.predictor_encoder(gt.unsqueeze(1))
                    )

                    F0_fake, N_fake = model.predictor.F0Ntrain(p_en, s_dur)

                    loss_dur = 0
                    for _s2s_pred, _text_input, _text_length in zip(
                        d, (d_gt), input_lengths
                    ):
                        _s2s_pred = _s2s_pred[:_text_length, :]
                        _text_input = _text_input[:_text_length].long()
                        _s2s_trg = torch.zeros_like(_s2s_pred)
                        for bib in range(_s2s_trg.shape[0]):
                            _s2s_trg[bib, : _text_input[bib]] = 1
                        _dur_pred = torch.sigmoid(_s2s_pred).sum(axis=1)
                        loss_dur += F.l1_loss(
                            _dur_pred[1 : _text_length - 1],
                            _text_input[1 : _text_length - 1],
                        )

                    loss_dur /= texts.size(0)

                    s = (
                        model.style_encoder(val_ref_style_mels.unsqueeze(1), val_ref_style_lengths)
                        if multispeaker
                        else model.style_encoder(gt.unsqueeze(1))
                    )

                    y_rec = model.decoder(en, F0_fake, N_fake, s)
                    loss_mel = stft_loss(y_rec.squeeze(1), wav.detach())

                    F0_real, _, F0 = model.pitch_extractor(gt.unsqueeze(1))

                    loss_F0 = F.l1_loss(F0_real, F0_fake) / 10

                    loss_test += (loss_mel).mean()
                    loss_align += (loss_dur).mean()
                    loss_f += (loss_F0).mean()

                    iters_test += 1
                except Exception as e:
                    log_warning("run into exception %s\n%s", e, traceback.format_exc())
                    continue

        epoch_duration = time.time() - start_time
        epoch_durations.append(epoch_duration)
        remaining_epochs = max(epochs - (epoch + 1), 0)
        eta = sum(epoch_durations) / len(epoch_durations) * remaining_epochs
        if iters_test == 0:
            log_error('Validation produced no successful batches; keeping training alive with val_loss=inf')
            loss_test = float('inf')
            iters_test = 1
        log_info(
            "ETA=%s Validation loss=%.6f, Dur loss=%.6f, F0 loss=%.6f"
            % (_format_eta(eta), loss_test / iters_test, loss_align / iters_test, loss_f / iters_test)
        )
        writer.add_scalar("eval/mel_loss", loss_test / iters_test, epoch)
        writer.add_scalar("eval/dur_loss", loss_align / iters_test, epoch)
        writer.add_scalar("eval/F0_loss", loss_f / iters_test, epoch)

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
            best_path = osp.join(log_dir, config.get("best_second_stage_path", "best_2nd.pth"))
            save_path = None
            if should_save_epoch:
                save_path = osp.join(log_dir, "epoch_2nd_%05d.pth" % epoch)
                atomic_torch_save(state, save_path)
            if is_best:
                best_loss = val_loss
                atomic_torch_save(state, best_path)
                log_info("New best Stage 2 validation loss: %.6f at epoch %d" % (best_loss, epoch))
            keep_paths = [best_path]
            if save_path:
                keep_paths.append(save_path)
            prune_checkpoints(
                log_dir,
                "epoch_2nd_*.pth",
                config.get("keep_last_checkpoints", 2),
                keep_paths=keep_paths,
            )

            # if estimate sigma, save the estimated sigma
            if diffusion_enabled and model_params.diffusion.dist.estimate_sigma_data:
                config["model_params"]["diffusion"]["dist"]["sigma_data"] = float(
                    np.mean(running_std)
                )

                with open(osp.join(log_dir, osp.basename(config_path)), "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=True)

        # ── Kokoro-faithful TensorBoard inference (every epoch) ───────────────
        _ = [model[key].eval() for key in model]
        _vp, _acoustic_norm, _prosodic_norm = extract_voicepack(
            model,
            root_path,
            eval_device,
            n_samples=200,
        )
        writer.add_scalar("voicepack/acoustic_norm", _acoustic_norm, epoch)
        writer.add_scalar("voicepack/prosodic_norm", _prosodic_norm, epoch)
        log_info(
            f"acoustic_norm={_acoustic_norm:.4f} prosodic_norm={_prosodic_norm:.4f}"
        )
        if _vp is not None and _test_tokens:
            _epoch_audio = run_kokoro_inference(
                model, _test_tokens, _vp, eval_device, text_cleaner
            )
            for i, (text, audio) in enumerate(_epoch_audio):
                writer.add_audio(
                    f"inference/test_{i + 1:02d}", audio, epoch, sample_rate=sr
                )
        if eval_device != device:
            _restore_training_devices()
            stft_loss = stft_loss.to(device)
        _set_train_mode(model)
        # ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
