import math

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)    
        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
    
    
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def _TPRLS_term(dr, dg):
    # Boolean-mask indexing has a data-dependent output shape, which compiles a
    # new MPS graph variant on every step; use a fixed-shape masked mean instead.
    tau = 0.04
    diff = dr - dg
    m_DG = torch.median(diff)
    mask = (diff < m_DG).to(diff.dtype)
    L_rel = (((diff - m_DG) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return tau - F.relu(tau - L_rel)

def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss += _TPRLS_term(dr, dg)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        loss += _TPRLS_term(dr, dg)
    return loss

def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


class GeneratorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(GeneratorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        out_device = y_hat.device
        mpd_device = _module_device(self.mpd)
        msd_device = _module_device(self.msd)

        y_mpd = y.to(mpd_device) if mpd_device is not None and y.device != mpd_device else y
        y_hat_mpd = y_hat.to(mpd_device) if mpd_device is not None and y_hat.device != mpd_device else y_hat
        y_msd = y.to(msd_device) if msd_device is not None and y.device != msd_device else y
        y_hat_msd = y_hat.to(msd_device) if msd_device is not None and y_hat.device != msd_device else y_hat

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y_mpd, y_hat_mpd, real_no_grad=True)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y_msd, y_hat_msd, real_no_grad=True)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g).to(out_device)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g).to(out_device)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        loss_gen_f = loss_gen_f.to(out_device)
        loss_gen_s = loss_gen_s.to(out_device)

        loss_rel = generator_TPRLS_loss(y_df_hat_r, y_df_hat_g).to(out_device) + generator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g).to(out_device)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, mpd, msd):
        super(DiscriminatorLoss, self).__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self, y, y_hat):
        out_device = y_hat.device
        mpd_device = _module_device(self.mpd)
        msd_device = _module_device(self.msd)

        y_mpd = y.to(mpd_device) if mpd_device is not None and y.device != mpd_device else y
        y_hat_mpd = y_hat.to(mpd_device) if mpd_device is not None and y_hat.device != mpd_device else y_hat
        y_msd = y.to(msd_device) if msd_device is not None and y.device != msd_device else y
        y_hat_msd = y_hat.to(msd_device) if msd_device is not None and y_hat.device != msd_device else y_hat

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y_mpd, y_hat_mpd)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y_msd, y_hat_msd)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        loss_rel = discriminator_TPRLS_loss(y_df_hat_r, y_df_hat_g).to(out_device) + discriminator_TPRLS_loss(y_ds_hat_r, y_ds_hat_g).to(out_device)


        d_loss = loss_disc_s.to(out_device) + loss_disc_f.to(out_device) + loss_rel
        
        return d_loss.mean()
   
    
class WavLMLoss(torch.nn.Module):

    def __init__(self, model, wd, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        # WavLM is used as a fixed perceptual feature extractor / SLM critic.
        # Freeze its weights to avoid parameter gradients and keep dropout disabled,
        # while still allowing gradients to flow through WavLM operations to y_rec.
        self.wavlm.requires_grad_(False)
        self.wavlm.eval()
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def train(self, mode: bool = True):
        super().train(mode)
        self.wavlm.eval()
        return self

    def _to_wavlm_device(self, *tensors):
        # WavLM may live on another device (wavlm_device); .to() keeps autograd
        # so gradients still reach generated audio on the training device.
        wavlm_device = _module_device(self.wavlm)
        return [t.to(wavlm_device) if wavlm_device is not None and t.device != wavlm_device else t for t in tensors]

    def forward(self, wav, y_rec):
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        if y_rec.ndim == 3:
            y_rec = y_rec.squeeze(1)
        out_device = y_rec.device
        wav, y_rec = self._to_wavlm_device(wav, y_rec)

        self.wavlm.eval()
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean().to(out_device)

    def generator(self, y_rec):
        if y_rec.ndim == 3:
            y_rec = y_rec.squeeze(1)
        out_device = y_rec.device
        (y_rec,) = self._to_wavlm_device(y_rec)
        self.wavlm.eval()
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1-y_df_hat_g)**2)

        return loss_gen.to(out_device)

    def discriminator(self, wav, y_rec):
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        if y_rec.ndim == 3:
            y_rec = y_rec.squeeze(1)
        out_device = y_rec.device
        wav, y_rec = self._to_wavlm_device(wav, y_rec)

        self.wavlm.eval()
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean().to(out_device)

    def discriminator_forward(self, wav):
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        out_device = wav.device
        (wav,) = self._to_wavlm_device(wav)
        self.wavlm.eval()
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs.to(out_device)


class GridFlickerLoss(torch.nn.Module):
    '''Penalise high-band level flicker locked to the decoder's frame grid (X21, J-305).

    The ISTFTNet decoder runs at 40 fps (600 samples) before its last block and 80 fps after it. Fine-tuned
    generators imprint that grid on fricative noise as phase-coherent 40/80/120/160 Hz modulation of the
    4-10 kHz log envelope ("bubbles"). For each carrier bin the log envelope of every active 600-sample cycle
    is projected onto the cycle's harmonics 1..4 and averaged across cycles; the squared magnitude of that
    average is the grid-coherent power. The loss is relu(power(y_rec) - power(wav)), so a crop is penalised only
    for flicker beyond the real recording's. Cycles are gated on the real crop (high-band share and level).
    Crops must start on the 600-sample grid (train_first.py crops at multiples of 600).
    '''

    def __init__(self, sr=24000, n_fft=128, hop=24, period=600, f_lo=4000.0, f_hi=10000.0, harmonics=4,
                 share_min=0.25, range_db=35.0, floor_db=-30.0):
        super().__init__()
        assert period % hop == 0
        self.n_fft, self.hop, self.period, self.k = n_fft, hop, period, period // hop
        self.share_min, self.range = share_min, 10 ** (-range_db / 10)
        self.floor_rel = 10 ** (floor_db / 10)
        f = torch.fft.rfftfreq(n_fft, 1 / sr)
        self.register_buffer('band', (f >= f_lo) & (f <= f_hi), persistent=False)
        self.register_buffer('hf', f >= f_lo, persistent=False)
        self.register_buffer('window', torch.hann_window(n_fft), persistent=False)
        # frame t of a cycle is centred at sample t*hop + n_fft/2 inside the cycle
        ph = 2 * math.pi * torch.arange(1, harmonics + 1)[:, None] * (torch.arange(self.k) * hop + n_fft / 2) / period
        self.register_buffer('cos', torch.cos(ph).T / self.k, persistent=False)   # [k, H]
        self.register_buffer('sin', torch.sin(ph).T / self.k, persistent=False)

    def _power(self, x):
        n = (x.size(-1) // self.period) * self.period
        x = x[..., :n]
        # center=False: frame t starts at sample t*hop, so its phase in the cycle is explicit; no padding, so
        # the last partial cycle is dropped rather than filled with zeros
        S = torch.stft(x, self.n_fft, self.hop, window=self.window, center=False, return_complex=True)
        return S.real ** 2 + S.imag ** 2  # [B, F, frames]

    def forward(self, y_rec, wav):
        y_rec = y_rec.reshape(y_rec.size(0), -1).float()
        wav = wav.reshape(wav.size(0), -1).float()
        P = self._power(y_rec)
        with torch.no_grad():
            Q = self._power(wav)
        B, C = y_rec.size(0), (min(P.size(-1), Q.size(-1)) // self.k)
        P, Q = P[..., :C * self.k], Q[..., :C * self.k]
        with torch.no_grad():
            tot = Q.sum(1).view(B, C, self.k).mean(-1)                       # [B, C]
            hfp = Q[:, self.hf].sum(1).view(B, C, self.k).mean(-1)
            gate = (hfp / tot.clamp_min(1e-12) > self.share_min) & (tot > tot.amax(1, keepdim=True) * self.range)
            g = gate.float()
            n = g.sum(1).clamp_min(1)                                         # active cycles per crop

            # floor relative to the real crop's band level: keeps log() and its gradient bounded in near-silent bins
            floor = (Q[:, self.band].mean((1, 2)) * self.floor_rel).clamp_min(1e-10)[:, None, None]

        def coherent(Pw):
            d = torch.log(Pw[:, self.band] + floor).view(B, -1, C, self.k)            # [B, bins, C, k]
            d = d - d.mean(-1, keepdim=True)
            zr, zi = d @ self.cos, d @ self.sin                                       # [B, bins, C, H]
            zr = (zr * g[:, None, :, None]).sum(2) / n[:, None, None]
            zi = (zi * g[:, None, :, None]).sum(2) / n[:, None, None]
            return (zr ** 2 + zi ** 2).sum(-1).mean(-1)                               # [B]

        with torch.no_grad():
            ref = coherent(Q)
        gen = coherent(P)
        use = (g.sum(1) >= 4).float()                                                 # need a few active cycles
        num, cnt = (F.relu(gen - ref) * use).sum(), use.sum().detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # DDP averages gradients over ranks; normalise by the global eligible-crop count so the result
            # equals the mean over eligible crops of the whole batch (Codex x21 review)
            cnt = cnt.clone()
            torch.distributed.all_reduce(cnt)
            return torch.distributed.get_world_size() * num / cnt.clamp_min(1)
        return num / cnt.clamp_min(1)
