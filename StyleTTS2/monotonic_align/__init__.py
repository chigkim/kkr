import numpy as np
import torch

from .core import maximum_path_c


def mask_from_lens(*args):
    if len(args) == 1:
        (lens,) = args
        max_len = int(lens.max().item()) if hasattr(lens, 'max') else int(max(lens))
        ids = torch.arange(max_len, device=lens.device if hasattr(lens, 'device') else None)
        return ids.unsqueeze(0) < lens.unsqueeze(1)

    if len(args) == 3:
        x, x_lens, y_lens = args
        b, t_x, t_y = x.shape
        x_ids = torch.arange(t_x, device=x.device).unsqueeze(0) < x_lens.to(x.device).unsqueeze(1)
        y_ids = torch.arange(t_y, device=x.device).unsqueeze(0) < y_lens.to(x.device).unsqueeze(1)
        return (x_ids.unsqueeze(2) & y_ids.unsqueeze(1)).to(dtype=x.dtype, device=x.device)

    raise TypeError(f'mask_from_lens expected 1 or 3 arguments, got {len(args)}')


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent_np = np.ascontiguousarray(neg_cent.detach().cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent_np.shape, dtype=np.int32))
    mask_np = mask.detach().cpu() if hasattr(mask, 'detach') else mask
    t_t_max = np.ascontiguousarray(mask_np.sum(1)[:, 0].numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask_np.sum(2)[:, 0].numpy().astype(np.int32))
    maximum_path_c(path, neg_cent_np, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
