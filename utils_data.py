import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import random


def get_subfolders(folder):
    # Get the full paths of the immediate subdirectories
    subfolders = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    return subfolders


# UBFC-rPPG
def UBFC_rPPG_LU_split(fold_num=5, fold_index=0):
    # split UBFC-rPPG dataset into training and testing parts
    # the function returns the file paths for the training set and test set.

    h5_dir = "./rPPGDataset/UBFC_rPPG/h5"
    train_list = []  # Train set
    val_list = []  # Test set

    val_subject = []  # TODO: Predefined test set subject list

    subject_path = "./rPPGDataset/UBFC_rPPG/dataset2"
    subjects = sorted([subject for subject in os.listdir(subject_path)])

    remove_list = ["subject11", "subject18", "subject20", "subject24"]
    for remove_subject in remove_list:
        if remove_subject in subjects:
            subjects.remove(remove_subject)

    for subject in subjects:
        h5_path = os.path.join(h5_dir, subject, "vid.h5")
        if os.path.isfile(h5_path):
            if subject in val_subject:
                val_list.append(h5_path)
            else:
                train_list.append(h5_path)

    return train_list, val_list


# PURE
def PURE_LU_split(fold_num=5, fold_index=0):
    # split PURE dataset into training and testing parts
    # the function returns the file paths for the training set and test set.

    h5_dir = "./rPPGDataset/PURE/h5"
    train_list = []  # Train set
    val_list = []  # Test set

    val_subject = []  # TODO: Predefined test set subject list

    subject_path = "./rPPGDataset/PURE/raw"
    subjects = sorted([subject for subject in os.listdir(subject_path)])

    remove_list = ["07-07", "07-02"]
    for remove_subject in remove_list:
        if remove_subject in subjects:
            subjects.remove(remove_subject)

    for subject in subjects:
        h5_path = os.path.join(h5_dir, subject, f"{subject}.h5")
        if os.path.isfile(h5_path):
            if subject in val_subject:
                val_list.append(h5_path)
            else:
                train_list.append(h5_path)

    return train_list, val_list


# BUAA
def BUAA_LU_split(fold_num=5, fold_index=0):
    # split BUAA dataset into training and testing parts
    # the function returns the file paths for the training set and test set.

    h5_dir = "./rPPGDataset/BUAA/h5"
    train_list = []  # Train set
    val_list = []  # Test set

    val_subject = []  # TODO: Predefined test set subject list

    subject_path = "./rPPGDataset/BUAA/raw"
    subjects = sorted([subject for subject in os.listdir(subject_path)])

    subject_folders = get_subfolders(h5_dir)
    for subject_folder in subject_folders:
        subject = os.path.basename(subject_folder)

        session_folders = get_subfolders(subject_folder)
        for session_folder in session_folders:
            session = os.path.basename(session_folder)

            if session not in ["lux_10.0", "lux_15.8", "lux_25.1", "lux_39.8", "lux_63.1", "lux_100.0"]:
                continue

            h5_paths = [os.path.join(session_folder, f) for f in os.listdir(session_folder) if f.endswith(".h5")]
            if not h5_paths:
                continue

            h5_path = h5_paths[0]
            if os.path.isfile(h5_path):
                if subject in val_subject:
                    val_list.append(h5_path)
                else:
                    train_list.append(h5_path)

    return train_list, val_list


class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list  # list of .h5 file paths for training
        self.T = T  # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], "r") as f:
            img_length = f["imgs"].shape[0]

            idx_start = np.random.choice(img_length - self.T)

            idx_end = idx_start + self.T

            img_seq = f["imgs"][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype("float32")
        return img_seq
