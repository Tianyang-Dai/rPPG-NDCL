import os
import h5py
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import random
import data_process.transforms as transforms
import cv2
from utils_phys import *


def get_subject_session(h5_path, dataset):
    if dataset == "UBFC-rPPG":
        subject = h5_path.split("/")[-2]
        return subject, None
    elif dataset == "PURE":
        subject_session = h5_path.split("/")[-1]
        subject_session = subject_session.split(".")[0]
        subject_session = subject_session.split("-")
        subject = subject_session[0]
        session = subject_session[1]
        return subject, session
    elif dataset in ["BUAA"]:
        subject = h5_path.split("/")[-3]
        session = h5_path.split("/")[-2]
        return subject, session


class Data_ND(Dataset):  # Noise-Disentangled Dataset
    def __init__(self, train_list, config, phase, state):
        self.train_list = train_list  # List of file paths for training .h5 files
        self.config = config
        self.phase = phase  # 'train' or 'test'
        self.state = state  # 'rppg' or 'head' or 'illum'
        self.T = config["T"]  # Video clip length
        self.H = config["image_size"]
        self.W = config["image_size"]
        self.speed_slow = 0.6
        self.speed_fast = 1.4
        self.max_idx = len(train_list)
        self.dataset = config["dataset"]

    def apply_transformations(self, img_seq, idcs, augment, idx):  # [T, H, W, C]
        speed = 1.0

        if augment not in ["frequency_augment", "temporal_augment"]:  # ! not Frequency/Temporal Augmentation
            img_seq = img_seq[idcs].transpose(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]

        # No Augmentation
        if augment == "no_augment":
            pass

        # Spatial Augmentation
        elif augment == "spatial_augment":
            # Random horizontal flip
            img_seq = transforms.augment_horizontal_flip(img_seq)
            # Random resize and crop
            img_seq = transforms.random_resized_crop(img_seq)

        # Temporal Augmentation)
        elif augment == "temporal_augment":
            # Random inversion, delay time
            img_seq = transforms.augment_time_reversal_delay(img_seq, idcs, self.T)

        # Frequency Augmentation
        elif augment == "frequency_augment":
            # Time resampling
            img_seq, idcs, speed = transforms.augment_speed(img_seq, idcs, self.T, self.speed_slow, self.speed_fast)

        # Color Augmentation
        elif augment == "color_augment":
            # Color jitter
            img_seq = transforms.augment_color_jitter(img_seq)

        # Inter-Instance Augmentation
        elif augment == "inter_instance_augment":
            subject, _ = get_subject_session(self.train_list[idx], self.dataset)
            while True:
                choices = [i for i in range(self.max_idx) if i != idx]
                random_choice = random.choice(choices)
                inter_subject, _ = get_subject_session(self.train_list[random_choice], self.dataset)
                if inter_subject != subject:
                    break
            with h5py.File(self.train_list[random_choice], "r") as f:
                img_length = f["imgs"].shape[0]  # f['imgs']: [T, H, W, C]
                idx_start = np.random.choice(img_length - self.T)
                idx_end = idx_start + self.T
                img_seq = f["imgs"][idx_start:idx_end]  # [T, H, W, C]
                img_seq = img_seq.transpose(3, 0, 1, 2)  # [C, T, H, W]

        else:
            # Illumination Augmentation
            if augment == "illumination_augment":
                # Illumination noise
                img_seq = transforms.augment_illumination_noise(img_seq)

            # Gaussian Augmentation
            elif augment == "gaussian_augment":
                # Add Gaussian noise to each pixel
                img_seq = transforms.augment_gaussian_noise(img_seq)

        img_seq = np.clip(img_seq, 0, 255)
        img_seq = img_seq / 255
        img_seq = torch.from_numpy(img_seq).float()  # [C, T, H, W]

        return img_seq, idcs, speed

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], "r") as f:
            img_length = f["imgs"].shape[0]  # f['imgs']: [T, H, W, C]
            idx_start = np.random.choice(img_length - self.T)

            if idx_start + int(self.T * 2.0) > img_length:
                idx_end = idx_start + self.T
                img_seq = f["imgs"][idx_start:idx_end].astype(np.float32)  # [T, H, W, C]
                bvp = f["bvp"][idx_start:idx_end]  # [T,]
            else:
                idx_end = idx_start + int(self.T * 2.0)
                img_seq = f["imgs"][idx_start:idx_end].astype(np.float32)  # [T, H, W, C]
                bvp = f["bvp"][idx_start:idx_end]  # [T,]

            idcs = np.arange(0, self.T, dtype=int)

            if self.state == "rppg":
                pos_augment_list = ["spatial_augment", "temporal_augment"]
                neg_augment_list = ["frequency_augment", "inter_instance_augment"]
            elif self.state == "head":
                pos_augment_list = ["color_augment"]
                neg_augment_list = ["frequency_augment", "inter_instance_augment"]
            elif self.state == "illum":
                pos_augment_list = ["spatial_augment"]
                neg_augment_list = ["frequency_augment", "color_augment"]

            # Anchor sample
            anc_vid, speed_idcs, speed = self.apply_transformations(img_seq=img_seq, idcs=idcs, augment="no_augment", idx=idx)
            # Positive sample
            pos_vid, speed_idcs, speed = self.apply_transformations(img_seq=img_seq, idcs=idcs, augment=random.choice(pos_augment_list), idx=idx)
            # Negative sample
            neg_vid, speed_idcs, speed = self.apply_transformations(img_seq=img_seq, idcs=idcs, augment=random.choice(neg_augment_list), idx=idx)

            wave = bvp[idcs]
            wave = wave - wave.mean()
            wave = wave / np.abs(wave).max()
            wave = torch.from_numpy(wave).float()
            hr = cal_hr(wave, 30)

            return {
                "anc_vid": anc_vid,  # [T, H, W, C]
                "pos_vid": pos_vid,  # [T, H, W, C]
                "neg_vid": neg_vid,  # [T, H, W, C]
                "bvp": wave,  # [T,]
                "hr": hr,  # []
            }

    def __len__(self):
        return len(self.train_list)
