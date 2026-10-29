import torch
import numpy as np
import torch.nn.functional as F


def _fit_audio(x, length):
    if length is None or length <= 0:
        return x
    length = int(length)
    if x.size(-1) > length:
        start = torch.randint(0, x.size(-1) - length + 1, (), device='cpu').item()
        return x[..., start:start + length]
    if x.size(-1) < length:
        return F.pad(x, (0, length - x.size(-1)))
    return x


def _fit_audio_pair(a, b, length):
    '''_fit_audio for two time-aligned clips, with one shared crop offset.'''
    if length is None or length <= 0:
        return a, b
    n = min(a.size(-1), b.size(-1))
    return _fit_audio(torch.cat([a[..., :n], b[..., :n]], dim=0), length).split([a.size(0), b.size(0)], dim=0)


class SLMAdversarialLoss(torch.nn.Module):
    def __init__(
        self,
        model,
        wl,
        sampler,
        min_len,
        max_len,
        batch_percentage=0.5,
        skip_update=10,
        sig=1.5,
        diffusion_enabled=True,
        fixed_mel_len=None,
        fixed_crop_size=None,
        conv_on_cpu=False,
        length_reg_ratio=0.75,
    ):
        super(SLMAdversarialLoss, self).__init__()
        self.model = model
        self.wl = wl
        self.sampler = sampler

        self.min_len = min_len
        self.max_len = max_len
        self.batch_percentage = batch_percentage

        self.sig = sig
        self.skip_update = skip_update
        self.diffusion_enabled = diffusion_enabled
        self.fixed_mel_len = fixed_mel_len
        self.fixed_crop_size = fixed_crop_size
        self.conv_on_cpu = conv_on_cpu
        self.length_reg_ratio = length_reg_ratio

    def forward(
        self,
        iters,
        y_rec_gt,
        y_rec_gt_pred,
        waves,
        mel_input_length,
        ref_text,
        ref_lengths,
        use_ind,
        s_trg,
        ref_s=None,
    ):
        # Build the mask at the padded/bucketed ref_text width, not
        # ref_lengths.max(): with fixed-width collation they differ, and the
        # narrower mask both breaks BERT/predictor shapes and reintroduces
        # per-batch dynamic widths on MPS.
        text_width = ref_text.size(1)
        text_mask = length_to_mask(ref_lengths, text_width).to(ref_text.device)
        bert_dur = self.model.bert(ref_text, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        if not self.diffusion_enabled:
            # Diffusion model was never trained — use ground-truth styles directly
            s_preds = s_trg
        elif use_ind and np.random.rand() < 0.5:
            s_preds = s_trg
        else:
            num_steps = np.random.randint(3, 5)
            if ref_s is not None:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur,
                    embedding_scale=1,
                    features=ref_s,  # reference from the same speaker as the embedding
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)
            else:
                s_preds = self.sampler(
                    noise=torch.randn_like(s_trg).unsqueeze(1).to(ref_text.device),
                    embedding=bert_dur,
                    embedding_scale=1,
                    embedding_mask_proba=0.1,
                    num_steps=num_steps,
                ).squeeze(1)


        s_dur = s_preds[:, 128:]
        s = s_preds[:, :128]

        d, _ = self.model.predictor(
            d_en,
            s_dur,
            ref_lengths,
            torch.randn(ref_lengths.shape[0], text_width, 2).to(ref_text.device),
            text_mask,
        )

        bib = 0

        output_lengths = []
        attn_preds = []

        # differentiable duration modeling
        conv_device = torch.device('cpu') if self.conv_on_cpu and ref_text.device.type == 'mps' else ref_text.device
        for _s2s_pred, _text_length in zip(d, ref_lengths):
            # Move at the fixed padded width, slice on the conv device: a
            # per-sample-length slice on MPS compiles one graph per length.
            _s2s_pred_org = _s2s_pred.to(conv_device)[:_text_length, :]

            _s2s_pred = torch.sigmoid(_s2s_pred_org)
            _dur_pred = _s2s_pred.sum(axis=-1)

            l = int(torch.round(_s2s_pred.sum()).item())
            l = max(l, 1)

            t = (
                torch.arange(0, l, device=conv_device)
                .unsqueeze(0)
                .expand((len(_s2s_pred), l))
            )
            loc = torch.cumsum(_dur_pred, dim=0) - _dur_pred / 2

            h = torch.exp(
                -0.5 * torch.square(t - (l - loc.unsqueeze(-1))) / (self.sig) ** 2
            )

            out = torch.nn.functional.conv1d(
                _s2s_pred_org.unsqueeze(0),
                h.unsqueeze(1),
                padding=h.shape[-1] - 1,
                groups=int(_text_length),
            )[..., :l]
            attn_preds.append(F.softmax(out.squeeze(), dim=0))

            output_lengths.append(l)


        # Bucket the predicted alignment width: max(output_lengths) is unique
        # nearly every step, and an exact-width attention/decoder graph would
        # compile a fresh MPS graph variant each time.
        max_len = ((max(output_lengths) + 63) // 64) * 64

        with torch.no_grad():
            t_en = self.model.text_encoder(ref_text, ref_lengths, text_mask)

        # Assemble on the conv device: the per-sample (text_len x pred_len)
        # copies have near-unique shapes every step (pred_len drifts with the
        # model), so on MPS they grow the graph cache without bound. One
        # bucketed fixed-shape transfer afterwards is cache-stable.
        s2s_attn = torch.zeros(
            len(ref_lengths), text_width, max_len, device=conv_device
        )
        for bib in range(len(output_lengths)):
            s2s_attn[bib, : ref_lengths[bib], : output_lengths[bib]] = attn_preds[bib]
        s2s_attn = s2s_attn.to(ref_text.device)

        asr_pred = t_en @ s2s_attn

        _, p_pred = self.model.predictor(d_en, s_dur, ref_lengths, s2s_attn, text_mask)

        if self.fixed_mel_len is not None:
            mel_len = int(self.fixed_mel_len)
        else:
            mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
            mel_len = min(mel_len, self.max_len // 2)

        # get clips

        sel = []
        pred_starts = []
        sp = []

        wav = []

        target_clips = max(2, int(np.ceil(self.batch_percentage * len(waves))))
        for bib in range(len(output_lengths)):
            mel_length_pred = output_lengths[bib]
            mel_length_gt = int(mel_input_length[bib].item() / 2)
            if mel_length_gt <= mel_len or mel_length_pred <= mel_len:
                continue

            sp.append(s_preds[bib])
            sel.append(bib)
            pred_starts.append(np.random.randint(0, mel_length_pred - mel_len))

            # get ground truth clips
            random_start = np.random.randint(0, mel_length_gt - mel_len)
            y = waves[bib][
                (random_start * 2) * 300 : ((random_start + mel_len) * 2) * 300
            ]
            wav.append(torch.from_numpy(y).to(dtype=torch.float32, device=ref_text.device))

            if len(wav) >= target_clips:  # prevent OOM due to longer lengths
                break

        if len(sp) <= 1:
            return None

        sp = torch.stack(sp)
        wav = torch.stack(wav).float()
        # Fixed-shape gather crops: narrow slices at random starts compile a
        # new MPS graph per offset; gather keeps one graph per (n, width).
        sel_idx = torch.tensor(sel, device=ref_text.device)
        starts = torch.tensor(pred_starts, device=ref_text.device)
        idx = (starts.unsqueeze(1) + torch.arange(mel_len, device=ref_text.device)).unsqueeze(1)
        en = torch.gather(
            asr_pred[sel_idx], -1, idx.expand(-1, asr_pred.size(1), -1)
        )
        p_en = torch.gather(
            p_pred[sel_idx], -1, idx.expand(-1, p_pred.size(1), -1)
        )

        F0_fake, N_fake = self.model.predictor.F0Ntrain(p_en, sp[:, 128:])
        y_pred = self.model.decoder(en, F0_fake, N_fake, sp[:, :128])

        fixed_crop_size = int(self.fixed_crop_size) if self.fixed_crop_size is not None else None
        if fixed_crop_size:
            wav = _fit_audio(wav, fixed_crop_size)
            y_pred = _fit_audio(y_pred, fixed_crop_size)
            # The recording and its reconstruction are compared frame by
            # frame below, so they must share one crop offset.
            y_rec_gt_fixed, y_rec_gt_pred_fixed = _fit_audio_pair(y_rec_gt, y_rec_gt_pred, fixed_crop_size)
        else:
            y_rec_gt_fixed = y_rec_gt
            y_rec_gt_pred_fixed = y_rec_gt_pred

        # discriminator loss
        if (iters + 1) % self.skip_update == 0:
            if np.random.randint(0, 2) == 0:
                wav = y_rec_gt_pred_fixed
                use_rec = True
            else:
                use_rec = False

            if use_rec:  # use reconstructed (shorter lengths), do length invariant regularization
                # Upstream crops the longer clip to the shorter one's length.
                # With fixed-size clips both lengths match, so that crop was
                # the whole clip and loss_reg was always 0; crop a fixed
                # shorter prefix of either clip instead (still one shape).
                if wav.size(-1) != y_pred.size(-1):
                    crop_real = wav.size(-1) > y_pred.size(-1)
                    crop_size = min(wav.size(-1), y_pred.size(-1))
                else:
                    crop_real = np.random.randint(0, 2) == 0
                    crop_size = max(1, int(wav.size(-1) * self.length_reg_ratio))
                full = wav if crop_real else y_pred
                real_GP = full[:, :, :crop_size]
                out_crop = self.wl.discriminator_forward(real_GP.detach().squeeze(1))
                out_org = self.wl.discriminator_forward(full.detach().squeeze(1))
                loss_reg = F.l1_loss(out_crop, out_org[..., : out_crop.size(-1)])

                if np.random.randint(0, 2) == 0:
                    real, fake = (real_GP, y_pred) if crop_real else (wav, real_GP)
                else:
                    real, fake = wav, y_pred
                d_loss = self.wl.discriminator(
                    real.detach().squeeze(1), fake.detach().squeeze(1)
                ).mean()

                # regularization (ignore length variation)
                d_loss += loss_reg

                out_gt = self.wl.discriminator_forward(y_rec_gt_fixed.detach().squeeze(1))
                out_rec = self.wl.discriminator_forward(
                    y_rec_gt_pred_fixed.detach().squeeze(1)
                )

                # regularization (ignore reconstruction artifacts)
                d_loss += F.l1_loss(out_gt, out_rec)

            else:
                d_loss = self.wl.discriminator(
                    wav.detach().squeeze(1), y_pred.detach().squeeze(1)
                ).mean()
        else:
            d_loss = 0

        # generator loss
        gen_loss = self.wl.generator(y_pred.squeeze(1))

        gen_loss = gen_loss.mean()

        return d_loss, gen_loss, y_pred.detach()


def length_to_mask(lengths, width=None):
    width = int(width) if width is not None else int(lengths.max())
    mask = (
        torch.arange(width)
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask
