from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

def run_text_aligner(text_aligner, mels, mel_lengths, texts, n_down, bucket=64):
    '''Run the text aligner on mels cut to the batch's longest clip.

    ASRCNN's GroupNorm normalises over the whole time axis, so the fixed-width
    MPS collation padding (1152 frames) changed which token each frame aligned
    to. The width is rounded up to `bucket` frames to keep the number of MPS
    graph shapes small, and the attention is zero-padded back to the full
    width so the fixed-shape code downstream is unchanged.

    Returns (s2s_pred, s2s_attn) with s2s_attn [B, text, mel_frames // 2**n_down].
    '''
    full = mels.size(-1)
    step = max(int(bucket), 2**n_down)
    width = min(full, -(-int(mel_lengths.max().item()) // step) * step)
    frames = mel_lengths.to(mels.device) // (2**n_down)
    mask = torch.arange(width // (2**n_down), device=mels.device).unsqueeze(0) >= frames.unsqueeze(1)
    _, s2s_pred, s2s_attn = text_aligner(mels[..., :width], mask, texts)
    pad = full // (2**n_down) - s2s_attn.size(-1)
    if pad > 0:
        s2s_attn = F.pad(s2s_attn, (0, pad))
    return s2s_pred, s2s_attn


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    