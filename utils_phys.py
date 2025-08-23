import torch
import math
import numpy as np
import pycwt
from scipy import signal


def cal_hr(output: torch.Tensor, Fs: float):
    """
    args:
        output: (1, T)
        Fs: sampling rate
    return:
        hr: heart rate
    """

    def compute_complex_absolute_given_k(output: torch.Tensor, k: torch.Tensor, N: int):
        two_pi_n_over_N = 2 * math.pi * torch.arange(0, N, dtype=torch.float) / N
        hanning = torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).view(1, -1)

        k = k.type(torch.FloatTensor)
        two_pi_n_over_N = two_pi_n_over_N
        hanning = hanning

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        return complex_absolute

    output = output.view(1, -1)

    N = output.size()[1]
    bpm_range = torch.arange(40, 180, dtype=torch.float)
    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.7, 4] Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)
    complex_absolute = (1.0 / complex_absolute.sum()) * complex_absolute
    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)  # max returns (values, indices)
    whole_max_idx = whole_max_idx.type(torch.float)  # The peak frequency of the power spectral density corresponds to the heart rate

    return whole_max_idx + 40  # Analogous to the Softmax operator


def maxscale(array1):
    list1 = []
    for i in range(array1.shape[0]):
        list1.append(np.mean(array1[i]))
    return np.array(list1)


def rowcal(array1, row1):
    array2 = np.zeros(array1.shape)
    for i in range(array1.shape[1]):
        array2[:, i] = array1[:, i] * row1
    return array2


frequencies = [3.75, 3.720703125, 3.69140625, 3.662109375, 3.6328125, 3.603515625, 3.5742187500000004, 3.544921875,
               3.515625, 3.486328125, 3.45703125, 3.427734375, 3.3984375, 3.369140625, 3.33984375, 3.310546875,
               3.28125,
               3.251953125, 3.22265625, 3.193359375, 3.1640625, 3.134765625, 3.10546875, 3.076171875, 3.046875,
               3.017578125,
               2.98828125, 2.958984375, 2.9296875, 2.9003906249999996, 2.87109375, 2.841796875, 2.8125, 2.783203125,
               2.75390625, 2.7246093750000004, 2.6953125, 2.666015625, 2.6367187500000004, 2.607421875, 2.578125,
               2.548828125, 2.51953125, 2.490234375, 2.4609375, 2.431640625, 2.40234375, 2.373046875, 2.34375,
               2.314453125,
               2.28515625, 2.255859375, 2.2265625, 2.197265625, 2.16796875, 2.138671875, 2.109375, 2.080078125,
               2.05078125,
               2.021484375, 1.9921875, 1.962890625, 1.93359375, 1.904296875, 1.875, 1.845703125, 1.81640625,
               1.7871093750000002, 1.7578125, 1.728515625, 1.69921875, 1.669921875, 1.640625, 1.611328125,
               1.58203125,
               1.552734375, 1.5234375, 1.494140625, 1.46484375, 1.435546875, 1.40625, 1.376953125, 1.34765625,
               1.3183593750000002, 1.2890625, 1.259765625, 1.23046875, 1.201171875, 1.171875, 1.142578125,
               1.11328125,
               1.083984375, 1.0546875, 1.025390625, 0.99609375, 0.966796875, 0.9375, 0.908203125, 0.87890625,
               0.849609375,
               0.8203125, 0.791015625, 0.76171875, 0.732421875, 0.703125, 0.673828125, 0.64453125]


def cwt_filtering(listin, samplingrate, frequencies=frequencies):
    sr = samplingrate
    plf1 = np.array(listin)
    result = pycwt.cwt(plf1, 1 / sr, freqs=np.array(frequencies))
    cwtmatr = result[0]
    scale1 = maxscale(abs(result[0]))
    co = np.argmax(scale1)
    myguasswindow = np.array([0.0 for x in range(len(scale1))])
    for j in range(len(scale1)):
        myguasswindow[j] = math.exp(-1 * ((j - co) / (0.08 * len(scale1))) ** 2)
    mycwtmatr = rowcal(abs(result[0]), myguasswindow)
    mycwtmatr2 = rowcal(result[0].real, myguasswindow)
    result_copy = result[1][:]
    result3 = pycwt.icwt(mycwtmatr2, result_copy, 1 / sr).real
    return result3, mycwtmatr, cwtmatr


def cwt_show(listin, sr):
    result = pycwt.cwt(listin, 1 / sr, freqs=np.array(frequencies))
    cwtmatr = abs(result[0])
    return cwtmatr


def peakcheckez(a, samplingrate):
    result = []
    for i in range(len(a)):
        if i == 0 or i == len(a) - 1:
            pass
        else:
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                result.append(i)

    hr_list = []
    if len(result) <= 1:
        hr = 0
    else:
        for i in range(len(result) - 1):
            hr = 60 * samplingrate / (result[i + 1] - result[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr


def eval_hr(tmp, samplingrate=30):
    f1 = 0.5
    f2 = 3
    samplingrate = samplingrate
    b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], "bandpass")
    tmp = signal.filtfilt(b, a, np.array(tmp))
    tmp = cwt_filtering(tmp, samplingrate)[0]

    hr_caled = peakcheckez(tmp, samplingrate)
    return hr_caled, tmp


def cxcorr_align(preds, labels):
    nom = torch.linalg.norm(preds, keepdim=True) * torch.linalg.norm(labels, keepdim=True)
    preds = preds.float()  # TODO
    labels = labels.float()  # TODO
    zi = torch.fft.irfft(torch.fft.rfft(preds) * torch.fft.rfft(labels.flip(-1)))
    cxcorr = zi / nom
    # cxcorr = cxcorr.abs()
    # out = np.abs(out)
    for b in range(cxcorr.shape[0]):
        _cxcorr = cxcorr[b]
        # max_idx = np.where(np.diff(np.sign(np.diff(_cxcorr))) < 0)[0] + 1
        # min_idx = np.where(np.diff(np.sign(np.diff(_cxcorr))) > 0)[0] + 1
        # torch
        max_idx = torch.where(torch.diff(torch.sign(torch.diff(_cxcorr))) < 0)[0] + 1
        min_idx = torch.where(torch.diff(torch.sign(torch.diff(_cxcorr))) > 0)[0] + 1

        # Check if min_idx and max_idx are empty
        if min_idx.numel() == 0 or max_idx.numel() == 0:
            print(f"Warning: min_idx or max_idx is empty for batch {b}.")
            continue  # Skip the current batch to avoid IndexError

        if min_idx[0] < max_idx[0]:
            # Needs to be shifted to the right
            shift = min_idx[0]
        else:
            # Needs to be shifted to the left
            shift = 0 - max_idx[0]
        preds[b] = torch.roll(preds[b], shift.item(), dims=0)

    return preds


def eval_metric(preds, labels):
    preds = np.array(preds).reshape(-1)  # Prediction
    labels = np.array(labels).reshape(-1)  # Ground truth
    temp = preds - labels
    ME = np.mean(temp)  # Mean error
    STD = np.std(temp)  # Standard deviation
    MAE = np.sum(np.abs(temp)) / len(temp)  # Mean Absolute Error
    RMSE = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))  # Root Mean Squared Error
    MER = np.mean(np.abs(temp) / labels)  # Mean Relative Error
    R = np.sum((preds - np.mean(preds)) * (labels - np.mean(labels))) / (
        0.01 + np.linalg.norm(preds - np.mean(preds), ord=2) * np.linalg.norm(labels - np.mean(labels), ord=2)
    )  # Pearson correlation coefficient
    return ME, STD, MAE, RMSE, MER, R
