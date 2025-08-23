import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import torchvision
import tqdm
import numpy as np
from data_process.mydataset import Data_ND
from itertools import cycle
import time
from utils_data import *
from utils_sig import *
from plot import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True, type=str)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--use_ddp", action="store_true", default=False)


def pretrain(config, rppg_train_loader, head_train_loader, illum_train_loader, model, logger, step):
    running_dic = None
    count = 0
    warm_up = config["warm_up"] if config["warm_up"] is not None else -1
    total_num = len(rppg_train_loader)

    rppg_train_loader = cycle(rppg_train_loader)
    head_train_loader = cycle(head_train_loader)
    illum_train_loader = cycle(illum_train_loader)

    # Loss
    rppg_loss = 0.0
    head_loss = 0.0
    illum_loss = 0.0
    # Metrics
    train_mae = 0.0
    train_rmse = 0.0
    train_r = 0.0
    train_ipr = 0.0

    for i in tqdm.tqdm(range(total_num)):
        # rPPG training
        data = next(rppg_train_loader)
        data["epoch"] = step
        data["state"] = "rppg"
        dic = model.optimize_parameters(data, config)
        rppg_loss += dic["rppg_loss"]
        train_mae += dic["train_mae"]
        train_rmse += dic["train_rmse"]
        train_r += dic["train_r"]
        train_ipr += dic["train_ipr"]

        if step > warm_up:  # Stage II
            # Head motion training
            data = next(head_train_loader)
            data["epoch"] = step
            data["state"] = "head"
            dic.update(model.optimize_parameters(data, config))
            head_loss += dic["head_loss"]

            # Illumination training
            data = next(illum_train_loader)
            data["epoch"] = step
            data["state"] = "illum"
            dic.update(model.optimize_parameters(data, config))
            illum_loss += dic["illum_loss"]

        count += 1

    rppg_loss /= count
    head_loss /= count
    illum_loss /= count
    train_mae /= count
    train_rmse /= count
    train_r /= count
    train_ipr /= count

    running_dic = {}
    running_dic["rppg_loss"] = rppg_loss
    running_dic["head_loss"] = head_loss
    running_dic["illum_loss"] = illum_loss
    running_dic["train_mae"] = train_mae
    running_dic["train_rmse"] = train_rmse
    running_dic["train_r"] = train_r
    running_dic["train_ipr"] = train_ipr

    txt = "epoch: {} / {}".format(step, config["epochs"])
    for k in list(running_dic.keys()):
        txt += "\n{}: {}".format(k, running_dic[k])
    print(txt)

    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

    return running_dic


def main(config, logger):
    exp_dir = os.path.join(config["result_dir"], config["dataset"])  # Store experimental records to the specified path, e.g. './results_test/PURE'
    os.makedirs(exp_dir, exist_ok=True)

    # Obtain training and testing file path lists by splitting the dataset
    if config["dataset"] == "UBFC-rPPG":
        train_list, test_list = UBFC_rPPG_LU_split(fold_num=5, fold_index=0)
    elif config["dataset"] == "PURE":
        train_list, test_list = PURE_LU_split(fold_num=5, fold_index=0)
    elif config["dataset"] == "BUAA":
        train_list, test_list = BUAA_LU_split(fold_num=5, fold_index=0)

    np.save(exp_dir + "/train_list.npy", train_list)  # Train set, e.g. './results_test/PURE/train_list.npy'
    np.save(exp_dir + "/test_list.npy", test_list)  # Test set, e.g. './results_test/PURE/test_list.npy'

    model = utils.create_model(config)

    rppg_train_dataset = Data_ND(train_list, config, phase="train", state="rppg")
    head_train_dataset = Data_ND(train_list, config, phase="train", state="head")
    illum_train_dataset = Data_ND(train_list, config, phase="train", state="illum")

    rppg_train_loader = DataLoader(
        rppg_train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_threads"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    head_train_loader = DataLoader(
        head_train_dataset,
        batch_size=config["small_batch_size"],
        num_workers=config["num_threads"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    illum_train_loader = DataLoader(
        illum_train_dataset,
        batch_size=config["small_batch_size"],
        num_workers=config["num_threads"],
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    if config["use_dwa"]:
        avg_cost = np.zeros([config["epochs"] + 1, 2], dtype=np.float32)
        dwa_t = config["dwa_T"]
        dwa_start_epoch = config["warm_up"] + 3 if config["warm_up"] != -1 else 2

    total_start_t = time.time()
    rppg_loss_epoch, head_loss_epoch, illum_loss_epoch = [], [], []
    train_mae_epoch, train_rmse_epoch, train_r_epoch, train_ipr_epoch = [], [], [], []
    best_SB_epoch_mae = 1  # Single branch
    best_MB_epoch_mae = config["warm_up"] + 1  # Multi-branch
    best_SB_epoch_ipr = 1  # Single branch
    best_MB_epoch_ipr = config["warm_up"] + 1  # Multi-branch
    best_SB_epoch_loss = 1  # Single branch
    best_MB_epoch_loss = config["warm_up"] + 1  # Multi-branch

    result_file = f"./results_train/{config['dataset']}/{config['experiment_name']}.txt"
    if not os.path.exists(result_file):
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    for step in range(config["start_epochs"], config["epochs"] + 1):
        epoch_start_t = time.time()  # Epoch start time

        if config["use_dwa"]:
            if step > dwa_start_epoch:
                head_w = avg_cost[step - 1, 0] / avg_cost[step - 2, 0]
                illum_w = avg_cost[step - 1, 1] / avg_cost[step - 2, 1]

                head_w_clipped = np.clip(head_w / dwa_t, -100, 100)  # Prevent overflow
                illum_w_clipped = np.clip(illum_w / dwa_t, -100, 100)  # Prevent overflow
                model.head_weight = 2 * np.exp(head_w_clipped) / (np.exp(head_w_clipped) + np.exp(illum_w_clipped))
                model.illum_weight = 2 * np.exp(illum_w_clipped) / (np.exp(head_w_clipped) + np.exp(illum_w_clipped))

                logger.add_scalar("head_weight", model.head_weight, step)
                logger.add_scalar("illum_weight", model.illum_weight, step)

        running_dic = pretrain(config, rppg_train_loader, head_train_loader, illum_train_loader, model, logger, step)

        rppg_loss_epoch.append(running_dic["rppg_loss"])
        head_loss_epoch.append(running_dic["head_loss"])
        illum_loss_epoch.append(running_dic["illum_loss"])
        train_mae_epoch.append(running_dic["train_mae"])
        train_rmse_epoch.append(running_dic["train_rmse"])
        train_r_epoch.append(running_dic["train_r"])
        train_ipr_epoch.append(running_dic["train_ipr"])

        # Plot loss vs. metrics line graph
        loss_dict = {
            "rppg_loss": rppg_loss_epoch,
            "head_loss": head_loss_epoch,
            "illum_loss": illum_loss_epoch,
        }
        metric_dict = {
            "train_mae": train_mae_epoch,
            "train_rmse": train_rmse_epoch,
            "train_r": train_r_epoch,
            "train_ipr": train_ipr_epoch,
        }
        title = "Loss and Metric Curve"
        dir = f"./imgs/{config['dataset']}"
        name = "pretrain"
        plot_loss_and_metric(loss_dict, metric_dict, title, dir, name)

        logger.add_scalar("rppg_loss_total", running_dic["rppg_loss"], step)
        logger.add_scalar("head_loss_total", running_dic["head_loss"], step)
        logger.add_scalar("illum_loss_total", running_dic["illum_loss"], step)
        logger.add_scalar("train_mae_total", running_dic["train_mae"], step)
        logger.add_scalar("train_rmse_total", running_dic["train_rmse"], step)
        logger.add_scalar("train_r_total", running_dic["train_r"], step)
        logger.add_scalar("train_r_total", running_dic["train_r"], step)
        logger.add_scalar("train_ipr_total", running_dic["train_ipr"], step)

        if config["use_dwa"]:
            avg_cost[step, 0] = running_dic["head_loss"]
            avg_cost[step, 1] = running_dic["illum_loss"]

        if step % config["save_epoch"] == 0:
            utils.save_checkpoint(
                {
                    "epoch": step,
                    "state_dict": model.model.state_dict(),
                },
                config,
            )

        epoch_end_t = time.time()
        epoch_t = epoch_end_t - epoch_start_t
        print(f"time/epoch: {epoch_t:.2f} s")

        # --------------------
        with open(result_file, "a") as f:
            f.write(f"\nEpoch: {step}\n")
            f.write(f"Time/Epoch: {epoch_t:.2f} s\n")
            f.write(f"rPPG Loss: {running_dic['rppg_loss']}\n")
            f.write(f"MAE: {running_dic['train_mae']}\n")
            f.write(f"RMSE: {running_dic['train_rmse']}\n")
            f.write(f"R: {running_dic['train_r']}\n")
            f.write(f"IPR: {running_dic['train_ipr']}\n")

        if step <= config["warm_up"]:
            if train_mae_epoch[step - 1] < train_mae_epoch[best_SB_epoch_mae - 1]:
                best_SB_epoch_mae = step
            if train_ipr_epoch[step - 1] < train_ipr_epoch[best_SB_epoch_ipr - 1]:
                best_SB_epoch_ipr = step
            if rppg_loss_epoch[step - 1] < rppg_loss_epoch[best_SB_epoch_loss - 1]:
                best_SB_epoch_loss = step

        else:
            if train_mae_epoch[step - 1] < train_mae_epoch[best_MB_epoch_mae - 1]:
                best_MB_epoch_mae = step
            if train_ipr_epoch[step - 1] < train_ipr_epoch[best_MB_epoch_ipr - 1]:
                best_MB_epoch_ipr = step
            if rppg_loss_epoch[step - 1] < rppg_loss_epoch[best_MB_epoch_loss - 1]:
                best_MB_epoch_loss = step
        # --------------------

    total_end_t = time.time()
    total_t = total_end_t - total_start_t
    print(f"total time: {total_t:.2f} s")

    # ------------------------------
    with open(result_file, "a") as f:
        f.write(f"\n[Single branch -> MAE]:\n")
        f.write(f"Best Epoch: {best_SB_epoch_mae}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_SB_epoch_mae-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_SB_epoch_mae-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_SB_epoch_mae-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_SB_epoch_mae-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_SB_epoch_mae-1]}\n")

        f.write(f"\n[Multi-branch -> MAE]:\n")
        f.write(f"Best Epoch: {best_MB_epoch_mae}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_MB_epoch_mae-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_MB_epoch_mae-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_MB_epoch_mae-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_MB_epoch_mae-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_MB_epoch_mae-1]}\n")

        f.write(f"\n[Single branch -> IPR]:\n")
        f.write(f"Best Epoch: {best_SB_epoch_ipr}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_SB_epoch_ipr-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_SB_epoch_ipr-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_SB_epoch_ipr-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_SB_epoch_ipr-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_SB_epoch_ipr-1]}\n")

        f.write(f"\n[Multi-branch -> IPR]:\n")
        f.write(f"Best Epoch: {best_MB_epoch_ipr}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_MB_epoch_ipr-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_MB_epoch_ipr-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_MB_epoch_ipr-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_MB_epoch_ipr-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_MB_epoch_ipr-1]}\n")

        f.write(f"\n[Single branch -> Loss]:\n")
        f.write(f"Best Epoch: {best_SB_epoch_loss}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_SB_epoch_loss-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_SB_epoch_loss-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_SB_epoch_loss-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_SB_epoch_loss-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_SB_epoch_loss-1]}\n")

        f.write(f"\n[Multi-branch -> Loss]:\n")
        f.write(f"Best Epoch: {best_MB_epoch_loss}\n")
        f.write(f"Best rPPG Loss: {rppg_loss_epoch[best_MB_epoch_loss-1]}\n")
        f.write(f"Best MAE: {train_mae_epoch[best_MB_epoch_loss-1]}\n")
        f.write(f"Best RMSE: {train_rmse_epoch[best_MB_epoch_loss-1]}\n")
        f.write(f"Best R: {train_r_epoch[best_MB_epoch_loss-1]}\n")
        f.write(f"Best IPR: {train_ipr_epoch[best_MB_epoch_loss-1]}\n")
    # ------------------------------


if __name__ == "__main__":
    opt = parser.parse_args()
    config = utils.read_config(opt.config_file)
    utils.init(config, opt.local_rank, opt.use_ddp)
    logger = SummaryWriter(
        log_dir=os.path.join(config["log_path"], config["dataset"], config["experiment_name"]),
        comment=config["experiment_name"],
    )
    main(config, logger)
    logger.close()
