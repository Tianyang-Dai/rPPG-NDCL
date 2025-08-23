import torch
import torch.nn as nn

tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft


class ContrastLoss(nn.Module):
    def __init__(self, config):
        super(ContrastLoss, self).__init__()
        self.config = config
        self.distance_func = nn.MSELoss()  # mean squared error for comparing two PSDs
        self.norm_psd = CalculateNormPSD(Fs=30, high_pass=20, low_pass=300)

    def compare_view_lists(self, list_a, list_p, list_n, temperature=0.07):
        B = len(list_a)
        M = 1
        anchor = torch.stack(list_a, dim=0)  # [B, 19]
        positive = torch.stack(list_p, dim=0)  # [B, 19]
        negative = torch.stack(list_n, dim=0)  # [B, 19]

        pos_dis = torch.exp(self.distance_func(anchor, positive) / temperature)
        neg_dis_total = 0

        for i in range(M):
            neg_dis = torch.exp(self.distance_func(anchor, negative) / temperature)
            neg_dis_total += neg_dis

        loss = torch.log10(pos_dis / neg_dis_total + 1)
        return loss

    def forward(self, model_output, cur_epoch, config, state):  # model_output: [B*5, T]
        if state == "rppg":
            B = model_output.shape[0] // 5
            samples = []
            for b in range(model_output.shape[0]):
                x = self.norm_psd(model_output[b])  # [19,]
                samples.append(x)  # [B*5, 19]

            branches = {}
            branches["anc"] = samples[0:B]  # Anchor sample, [B, 19]
            branches["pos"] = samples[B : B * 2]  # Positive sample, [B, 19]
            branches["neg"] = samples[B * 2 : B * 3]  # Negative sample, [B, 19]
            branches["pseudo_pos"] = samples[B * 3 : B * 4]  # Pseudo-positive sample, [B, 19]
            branches["pseudo_neg"] = samples[B * 4 : B * 5]  # Pseudo-negative sample, [B, 19]

            rppg_loss = self.compare_view_lists(branches["anc"], branches["pos"], branches["neg"])  # rPPG-rPPG loss
            pseudo_loss = self.compare_view_lists(branches["anc"], branches["pseudo_pos"], branches["pseudo_neg"])  # Pseudo-rPPG loss

            if cur_epoch <= config["warm_up"]:
                factor = cur_epoch / config["warm_up"]
                contrast_loss = factor * rppg_loss + (1 - factor) * pseudo_loss
            else:
                contrast_loss = rppg_loss

        elif state in ["head", "illum"]:
            B = model_output.shape[0] // 3
            samples = []
            for b in range(model_output.shape[0]):
                x = self.norm_psd(model_output[b])  # [19,]
                samples.append(x)  # [B*3, 19]

            branches = {}
            branches["anc"] = samples[0:B]  # Anchor sample, [B, 19]
            branches["pos"] = samples[B : B * 2]  # Positive sample, [B, 19]
            branches["neg"] = samples[B * 2 : B * 3]  # Negative sample, [B, 19]

            rppg_loss = self.compare_view_lists(branches["anc"], branches["pos"], branches["neg"])  # rPPG-rPPG loss

            contrast_loss = rppg_loss

        return contrast_loss


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent.
    # "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video."
    # Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), "constant", 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm="forward"))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x
