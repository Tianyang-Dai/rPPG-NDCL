import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def resample_clip(video, length):
    video = np.transpose(video, (3, 0, 1, 2)).astype(float)
    video = interpolate_clip(video, length)
    video = np.transpose(video, (1, 2, 3, 0))
    return video


def prepare_clip(clip):
    clip = np.transpose(clip, (3, 0, 1, 2))  # [C, T, H, W]
    clip = clip.astype(np.float64)
    return clip


def augment_speed(clip, idcs, frames_per_clip, speed_slow, speed_fast):
    """Interpolates clip to frames_per_clip length given slicing indices, which
    can be floats.
    """
    vid_len = len(clip)
    within_bounds = False
    while not within_bounds:
        # speed_c = np.random.uniform(speed_slow, speed_fast)
        speed_c = np.random.choice([0.6, 0.8, 1.2, 1.4])  # A random element from the list
        min_idx = idcs[0].astype(int)
        max_idx = np.round(frames_per_clip * speed_c + min_idx).astype(int)
        if max_idx < vid_len:
            within_bounds = True
    speed_c = (max_idx - min_idx) / frames_per_clip  # accomodate rounding of end-indices
    clip = clip[min_idx:max_idx]
    clip = prepare_clip(clip)
    interped_clip = interpolate_clip(clip, frames_per_clip)
    interped_idcs = np.linspace(min_idx, max_idx - 1, frames_per_clip)
    return interped_clip, interped_idcs, speed_c


def interpolate_clip(clip, length):
    """
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    """
    C, T, H, W = clip.shape
    clip = torch.from_numpy(clip[np.newaxis])
    clip = F.interpolate(clip, (length, H, W), mode="trilinear", align_corners=True)
    return clip[0].numpy()


def resize_clip(clip, length):
    """
    Input:
        clip: numpy array of shape [C,T,H,W]
        length: number of time points in output interpolated sequence
    Returns:
        Tensor of shape [C,T,H,W]
    """
    T = clip.shape[1]
    clip = torch.from_numpy(np.ascontiguousarray(clip[np.newaxis]))
    clip = F.interpolate(clip, (T, length, length), mode="trilinear", align_corners=False)
    return clip[0].numpy()


def random_resized_crop(clip, crop_scale_lims=[0.5, 0.75]):
    """Randomly crop a subregion of the video and resize it back to original size.
    Arguments:
        clip (np.array): expects [C,T,H,W]
    Returns:
        clip (np.array): same dimensions as input
    """
    C, T, H, W = clip.shape
    crop_scale = np.random.uniform(crop_scale_lims[0], crop_scale_lims[1])
    crop_length = np.round(crop_scale * H).astype(int)
    crop_start_lim = H - (crop_length)
    x1 = np.random.randint(0, crop_start_lim + 1)
    y1 = x1
    x2 = x1 + crop_length
    y2 = y1 + crop_length
    cropped_clip = clip[:, :, y1:y2, x1:x2]
    resized_clip = resize_clip(cropped_clip, H)
    return resized_clip


def augment_gaussian_noise(clip):
    clip = clip + np.random.normal(0, 2, clip.shape)
    return clip


def augment_illumination_noise(clip):
    clip = clip + np.random.normal(0, 10)
    return clip


def augment_time_reversal_delay(clip, idcs, frames_per_clip):
    # clip: [T, H, W, C]
    if np.random.rand() > 0.5:
        clip = np.flip(clip, 0)

    vid_len = len(clip)
    min_idx = idcs[0] + np.random.randint(0, vid_len - frames_per_clip + 1)
    max_idx = min_idx + frames_per_clip
    assert max_idx <= vid_len
    clip = clip[min_idx:max_idx]
    clip = prepare_clip(clip)
    return clip


def augment_horizontal_flip(clip):
    clip = np.flip(clip, 3)
    return clip


def augment_color_jitter(clip):
    clip = clip.astype(np.uint8)  # [C, T, H, W]
    colored_clip = np.zeros_like(clip)
    T = clip.shape[1]

    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    for t in range(T):
        frame = clip[:, t, :, :]  # [C, H, W]
        frame = np.transpose(frame, (1, 2, 0))  # [H, W, C]
        frame = Image.fromarray(frame)  # Convert to PIL image
        colored_frame = color_jitter(frame)
        colored_clip[:, t, :, :] = np.transpose(np.array(colored_frame), (2, 0, 1))  # [C, H, W]
    return colored_clip
