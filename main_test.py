import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import tqdm
import numpy as np
import time
from models.networks import PhysNet as backbone
import h5py
import os
from plot import *
from utils_data import *
from utils_phys import *
from utils_sig import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_file", required=True, type=str)
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--use_ddp", action="store_true", default=False)


def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp)) / len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr)) * (HR_rel - np.mean(HR_rel))) / (0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    return me, std, mae, rmse, mer, p


def test(config, model, test_list, test_dir, frame_interval=300):
    @torch.no_grad()
    def dl_model(imgs_clip):
        # Model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3, 0, 1, 2))
        img_batch = img_batch[np.newaxis].astype("float32")
        img_batch = torch.tensor(img_batch).cuda()  # [1, C, T, H, W]

        X_rppg, _, _, _, _, _ = model(img_batch)
        rppg = X_rppg  # [1, T]
        rppg = rppg.detach().cpu().numpy()
        return rppg

    for h5_path in test_list:
        h5_path = str(h5_path)

        with h5py.File(h5_path, "r") as f:
            imgs = f["imgs"]
            bvp = f["bvp"]
            fs = config["fs"]

            duration = np.min([imgs.shape[0], bvp.shape[0]])
            num_blocks = int(duration // frame_interval)

            rppg_list = []
            bvp_list = []

            for b in range(num_blocks):
                rppg_clip = dl_model(imgs[b * frame_interval : (b + 1) * frame_interval])
                rppg_list.append(rppg_clip)
                bvp_list.append(bvp[b * frame_interval : (b + 1) * frame_interval])

            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)
            results = {"rppg_list": rppg_list, "bvp_list": bvp_list}

            # TODO
            if config["dataset"] == "UBFC-rPPG":
                np.save(test_dir + "/" + h5_path.split("/")[-2], results)
            elif config["dataset"] == "PURE":
                np.save(test_dir + "/" + h5_path.split("/")[-1][:-3], results)
            elif config["dataset"] == "BUAA":
                np.save(test_dir + "/" + h5_path.split("/")[-1][:-3], results)

    file_extension = ".npy"
    file_list = [file for file in os.listdir(test_dir) if file.endswith(file_extension)]  # Get list of .npy files
    file_list.sort()  # Sort the file list in order

    hr_rppg_list = []
    hr_bvp_list = []

    for file in file_list:
        # Load .npy file
        file_path = os.path.join(test_dir, file)
        data = np.load(file_path, allow_pickle=True)

        rppg_data = data.item()["rppg_list"]  # [N, T]
        bvp_data = data.item()["bvp_list"]  # [N, T]
        N, T = rppg_data.shape

        y_rppg = butter_bandpass(rppg_data, 0.6, 4, fs)  # [N*T,]
        y_bvp = butter_bandpass(bvp_data, 0.6, 4, fs)  # [N*T,]
        y_rppg = y_rppg.reshape((N, T))  # [N, T]
        y_bvp = y_bvp.reshape((N, T))  # [N, T]

        for n in range(N):
            y_rppg_temp = torch.from_numpy(y_rppg[n].copy())
            y_bvp_temp = torch.from_numpy(y_bvp[n].copy())
            hr_rppg, _ = eval_hr(y_rppg_temp, 30)
            hr_bvp, _ = eval_hr(y_bvp_temp, 30)

            hr_rppg_list.append(hr_rppg)
            hr_bvp_list.append(hr_bvp)

    _, _, mae, rmse, _, r = MyEval(hr_rppg_list, hr_bvp_list)

    result_file = f"{config['result_dir']}/test_{config['dataset']}.txt"
    if not os.path.exists(result_file):
        os.makedirs(os.path.dirname(result_file), exist_ok=True)

    with open(result_file, "a") as f:
        f.write(f"{config['load_model']}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R: {r}\n")

    print("\n")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"R: {r}")

    # --------------------
    demo_dir = "./demo/results"
    os.makedirs(demo_dir, exist_ok=True)
    fps = 30
    y_rppg_plot = y_rppg[0]
    y_bvp_plot = y_bvp[0]

    # Prediction
    hr, psd_y, psd_x = hr_fft(y_rppg_plot, fs=fps)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

    ax1.plot(np.arange(len(y_rppg_plot)) / fps, y_rppg_plot)
    ax1.set_xlabel("time (sec)")
    ax1.grid("on")
    ax1.set_title("rPPG waveform")

    ax2.plot(psd_x, psd_y)
    ax2.set_xlabel("heart rate (bpm)")
    ax2.set_xlim([40, 200])
    ax2.grid("on")
    ax2.set_title("PSD")

    fig.suptitle("Prediction", fontsize=16)  # Set overall title
    pred_path = os.path.join(demo_dir, f"{config['dataset']}_pred.png")
    if os.path.exists(pred_path):  # If the file exists, delete it
        os.remove(pred_path)
    plt.savefig(pred_path)
    print("Prediction heart rate: %.2f bpm" % hr)

    # Ground truth
    hr, psd_y, psd_x = hr_fft(y_bvp_plot, fs=fps)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

    ax1.plot(np.arange(len(y_bvp_plot)) / fps, y_bvp_plot)
    ax1.set_xlabel("time (sec)")
    ax1.grid("on")
    ax1.set_title("rPPG waveform")
    ax2.plot(psd_x, psd_y)
    ax2.set_xlabel("heart rate (bpm)")
    ax2.set_xlim([40, 200])
    ax2.grid("on")
    ax2.set_title("PSD")

    fig.suptitle("Ground truth", fontsize=16)  # Set overall title
    gt_path = os.path.join(demo_dir, f"{config['dataset']}_gt.png")
    if os.path.exists(gt_path):  # If the file exists, delete it
        os.remove(gt_path)
    plt.savefig(gt_path)
    print("Ground truth heart rate: %.2f bpm" % hr)
    # --------------------


def main(config, logger):
    exp_dir = os.path.join(config["result_dir"], config["dataset"])  # Store experimental records to the specified path, e.g. './results/PURE'
    test_dir = os.path.join(exp_dir, "test")  # e.g. './results/PURE/test'
    os.makedirs(test_dir, exist_ok=True)

    # Load test file paths
    test_list = list(np.load(f"{exp_dir}/test_list.npy"))

    model = backbone(config)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(config["load_model"])["state_dict"])
    model.eval()

    frame_interval = config["test_T"] * config["fs"]  # Obtain rPPG for a 30-s video clip

    test(config, model, test_list, test_dir, frame_interval)


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
