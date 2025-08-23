import numpy as np
import torch
import time
import math
from models.BaseModel import BaseModel
from models.networks import *
from models.networks import PhysNet as backbone
from loss import ContrastLoss
from utils_phys import *
from IrrelevantPowerRatio import IrrelevantPowerRatio

# Pseudo-label
from pseudo_label.methods.CHROME_DEHAAN import *
from pseudo_label.methods.GREEN import *
from pseudo_label.methods.ICA_POH import *
from pseudo_label.methods.LGI import *
from pseudo_label.methods.PBV import *
from pseudo_label.methods.POS_WANG import *
from pseudo_label.methods.OMIT import *


def get_pseudo_label(pos_vid, neg_vid, fs):
    pos_vid = pos_vid.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
    neg_vid = neg_vid.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
    B = pos_vid.shape[0]

    pos_rPPG_pseudo = []
    neg_rPPG_pseudo = []
    for b in range(B):  # T = 144
        pos_rPPG_pseudo_temp = CHROME_DEHAAN(pos_vid[b, :, :, :, :])  # [T,]
        neg_rPPG_pseudo_temp = CHROME_DEHAAN(neg_vid[b, :, :, :, :])  # [T,]
        pos_rPPG_pseudo.append(pos_rPPG_pseudo_temp)
        neg_rPPG_pseudo.append(neg_rPPG_pseudo_temp)

    pos_rPPG_pseudo = np.array(pos_rPPG_pseudo)  # [B, T]
    neg_rPPG_pseudo = np.array(neg_rPPG_pseudo)  # [B, T]
    rPPG_pseudo = np.concatenate((pos_rPPG_pseudo, neg_rPPG_pseudo), axis=0)  # [B*2, T]

    rPPG_pseudo = torch.from_numpy(rPPG_pseudo).float()
    HR_pseudo = [cal_hr(rPPG_pseudo[i : i + 1], 30) for i in range(B * 2)]
    HR_pseudo = torch.tensor(HR_pseudo).unsqueeze(1)  # [B*2, 1]
    rPPG_pseudo = rPPG_pseudo  # [B*2, T]
    return rPPG_pseudo, HR_pseudo


class Mine(nn.Module):  # Mutual information neural estimaiton
    def __init__(self, output_dim, reduc_factor):
        super(Mine, self).__init__()
        self.output_dim = output_dim
        self.reduc_factor = reduc_factor
        self.fc1_x = nn.Linear(int(output_dim / reduc_factor), int(output_dim / reduc_factor / 2))
        self.fc1_y = nn.Linear(int(output_dim / reduc_factor), int(output_dim / reduc_factor / 2))
        self.fc2 = nn.Linear(int(output_dim / reduc_factor / 2), 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class OrthLoss(nn.Module):  # Mutual information loss
    def __init__(self):
        super(OrthLoss, self).__init__()
        self.mi_coeff = 0.0001
        self.output_dim = 64
        self.reduc_factor = 1
        self.MI = Mine(self.output_dim, self.reduc_factor)

    def mi_estimator(self, x, y, y_):
        joint, marginal = self.MI(x, y), self.MI(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def forward(self, input1, input2):
        input2_shuffle = torch.index_select(input2, 0, torch.randperm(input2.shape[0]).cuda())
        MI_act_input2 = self.mi_estimator(input1, input2, input2_shuffle)
        MI = 0.25 * (MI_act_input2) * self.mi_coeff
        orth_loss = MI
        return orth_loss


class ND(BaseModel):  # Noise disentangling
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.batch_size = config["batch_size"]
        self.warm_up_epoch = config["warm_up"] if config["warm_up"] is not None else -1
        self.model = backbone(config)

        if config["continue_train"]:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config["load_model"])["state_dict"])
            print("load continue model!")
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.orth_loss_func = OrthLoss().cuda()  # Mutual information loss
        self.contrast_loss_func = ContrastLoss(config)  # Contrastive loss
        self.IPR = IrrelevantPowerRatio(Fs=config["fs"], high_pass=40, low_pass=250)  # Irrelevant power ratio
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config["lr"]))

        self.head_weight = 1.0
        self.illum_weight = 1.0

    def optimize_parameters(self, data, config):
        self.model.train()

        cur_epoch = data["epoch"]
        state = data["state"]
        B = self.batch_size
        log_dic = {}

        if state == "rppg":
            X_rppg, X_head, X_illum, x_fc_rppg, x_fc_head, x_fc_illum = self.forward(data, config)  # [B*3, T]*3, [B*3, 64]*3

            # Pseudo-label
            pos_vid = data["pos_vid"].detach().cpu().numpy()  # [B, C, T, H, W]
            neg_vid = data["neg_vid"].detach().cpu().numpy()  # [B, C, T, H, W]
            rPPG_pseudo, HR_pseudo = get_pseudo_label(pos_vid, neg_vid, config["fs"])
            rPPG_pseudo = rPPG_pseudo.cuda()  # [B*2, T]
            HR_pseudo = HR_pseudo.cuda()  # [B*2, 1]

            # Contrastive loss
            signal = torch.cat((X_rppg, rPPG_pseudo), dim=0)  # [B*5, 144]
            contrast_loss = self.contrast_loss_func(signal, cur_epoch, config, state)
            loss = contrast_loss

            # Metrics
            # normalization
            rPPG_peak = X_rppg[:B]  # [B, T]
            rPPG = (rPPG_peak - rPPG_peak.mean(dim=-1, keepdim=True)) / torch.abs(rPPG_peak).max(dim=-1, keepdim=True).values

            ipr = torch.mean(self.IPR(X_rppg[:B].clone().detach()))

            rPPG_gt = data["bvp"].cuda()  # Ground truth rPPG
            if cur_epoch > 1:
                rPPG = cxcorr_align(rPPG, rPPG_gt)  # Alignment
            HR_temp = []
            HR_gt_temp = []
            rPPG_temp = rPPG.detach().cpu().numpy()
            rPPG_gt_temp = rPPG_gt.detach().cpu().numpy()
            rPPG_temp = torch.from_numpy(rPPG_temp).float()
            rPPG_gt_temp = torch.from_numpy(rPPG_gt_temp).float()
            HR = [cal_hr(rPPG_temp[i : i + 1], 30) for i in range(B)]
            HR_gt = [cal_hr(rPPG_gt_temp[i : i + 1], 30) for i in range(B)]
            HR_temp.extend(HR)
            HR_gt_temp.extend(HR_gt)
            _, _, MAE, RMSE, _, R = eval_metric(HR_temp, HR_gt_temp)

            log_dic = {
                "rppg_loss": loss.item(),
                "train_mae": MAE,
                "train_rmse": RMSE,
                "train_r": R,
                "train_ipr": ipr.item(),
            }

        elif state == "head":
            X_rppg, X_head, X_illum, x_fc_rppg, x_fc_head, x_fc_illum = self.forward(data, config)  # [B*3, T]*3, [B*3, 64]*3
            noise = X_head  # [B*3, T]

            contrast_loss = self.contrast_loss_func(noise, cur_epoch, config, state)  # Contrastive loss
            orth_loss = self.orth_loss_func(x_fc_rppg, x_fc_head) + self.orth_loss_func(x_fc_rppg, x_fc_illum)  # Mutual information loss
            loss = (contrast_loss + orth_loss) * self.head_weight

            log_dic = {"head_loss": loss.item()}

        elif state == "illum":
            X_rppg, X_head, X_illum, x_fc_rppg, x_fc_head, x_fc_illum = self.forward(data, config)  # [B*3, T]*3, [B*3, 64]*3
            noise = X_illum  # [B*3, T]

            contrast_loss = self.contrast_loss_func(noise, cur_epoch, config, state)  # Contrastive loss
            orth_loss = self.orth_loss_func(x_fc_rppg, x_fc_head) + self.orth_loss_func(x_fc_rppg, x_fc_illum)  # Mutual information loss
            loss = (contrast_loss + orth_loss) * self.illum_weight

            log_dic = {"illum_loss": loss.item()}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return log_dic

    def forward(self, data, config):
        anc_vid = data["anc_vid"]  # [B, C, T, H, W]
        pos_vid = data["pos_vid"]  # [B, C, T, H, W]
        neg_vid = data["neg_vid"]  # [B, C, T, H, W]
        vid = torch.cat([anc_vid, pos_vid, neg_vid], dim=0).cuda()  # [B*3, C, T, H, W]
        return self.model(x=vid)
