from math import floor, ceil

import numpy as np
import torch
from pywt import DiscreteContinuousWavelet, integrate_wavelet
from pywt._cwt import next_fast_len

fftmodule = torch.fft


def cwt1d(data, scales, int_psi_scales, axis=-1, out_dtype='real', device='cpu'):
    if out_dtype == 'complex':
        out = torch.empty((np.size(scales),) + data.shape, dtype=torch.complex64, device=device)
    else:
        out = torch.empty((np.size(scales),) + data.shape, device=device)
    if data.ndim > 1:
        # move axis to be transformed last (so it is contiguous)
        data = data.swapaxes(-1, axis)

        # reshape to (n_batch, data.shape[-1])
        data_shape_pre = data.shape
        data = data.reshape((-1, data.shape[-1]))
    size_scale0 = -1
    fft_data = None

    if int_psi_scales is not None:
        for i, (int_psi_scale, scale) in enumerate(zip(int_psi_scales, scales)):
            size_scale = next_fast_len(
                data.shape[-1] + int_psi_scale.size()[0] - 1
            )

            if size_scale != size_scale0:
                # Must recompute fft_data when the padding size changes.
                fft_data = fftmodule.fft(data, size_scale, axis=-1)
            size_scale0 = size_scale
            fft_wav = fftmodule.fft(int_psi_scale, size_scale, axis=-1)
            conv = fftmodule.ifft(fft_wav * fft_data, axis=-1)
            conv = conv[..., :data.shape[-1] + int_psi_scale.size()[0] - 1]

            coef = - torch.sqrt(torch.tensor(scale)) * torch.diff(conv)

            if out_dtype == 'real':
                coef = coef.real
            # transform axis is always -1 due to the data reshape above
            d = (coef.shape[-1] - data.shape[-1]) / 2.
            if d > 0:
                coef = coef[..., floor(d):-ceil(d)]
            elif d < 0:
                raise ValueError(
                    "Selected scale of {} too small.".format(scale))
            if data.ndim > 1:
                # restore original data shape and axis position
                coef = coef.reshape(data_shape_pre)
                coef = coef.swapaxes(axis, -1)
            if out_dtype == 'real':
                out[i, ...] = coef.real
            else:
                out[i, ...] = coef
    if len(out.shape) == 4:
        out = out.permute(1, 2, 0, 3)
    return out


def generate_int_psi_scales(scales, wavelet, device):
    wavelet = DiscreteContinuousWavelet(wavelet)
    precision = 12
    int_psi, x = integrate_wavelet(wavelet, precision=precision)
    int_psi_scales = []
    for i, scale in enumerate(scales):
        step = x[1] - x[0]
        j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
        j = j.astype(int)  # floor
        if j[-1] >= int_psi.size:
            j = np.extract(j < int_psi.size, j)
        int_psi_scale = int_psi[j][::-1]
        int_psi_scales.append(torch.tensor(int_psi_scale.copy()).to(device))
    return int_psi_scales


def get_representation(data):
    if data.dtype != torch.complex64:
        print('Data is not complex!')
        return

    input_shape = data.shape
    data = data.view(-1, input_shape[2], input_shape[3])
    amp = torch.abs(data)
    phase = torch.angle(data)
    H = phase / (2 * torch.pi)
    L = 1 - 2 ** (-amp)
    S = torch.ones_like(amp)

    repr = torch.stack([H, L, S], dim=1)  # now (n_batch, 3, 80, 200)

    # Use the custom PyTorch function for HSL to RGB conversion
    repr_rgb = hls_to_rgb_pytorch(repr[:, 0], repr[:, 1], repr[:, 2])

    return repr_rgb.permute(0, 3, 1, 2)  # Adjust the dimensions as needed


def hls_to_rgb_pytorch(h, l, s):
    # Ensure the input tensors have the same shape
    assert h.shape == l.shape == s.shape

    # Reshape the input tensors to (batch_size, height, width, channels)
    h = h.unsqueeze(-1)
    l = l.unsqueeze(-1)
    s = s.unsqueeze(-1)

    # Calculate RGB values using PyTorch operations
    c = (1 - torch.abs(2 * l - 1)) * s
    x = c * (1 - torch.abs((h / 60.0) % 2 - 1))
    m = l - 0.5 * c

    if (0 <= h < 60) or (300 <= h < 360):
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        raise ValueError("Invalid h value")

    # Add m to each channel
    r, g, b = r + m, g + m, b + m

    return torch.stack([r, g, b], dim=-1)


coefficient_mexh = 2 / (np.sqrt(3) * np.power(np.pi, 0.25))


def mexh(t):
    exponent = np.exp(-t ** 2 / 2)
    polynomial = 1 - t ** 2
    return coefficient_mexh * exponent * polynomial


def integrate(section_num, scales, precision=12, diapason=(-8, 8), func=mexh):
    points_num = 2 ** precision
    diapason_len = diapason[1] - diapason[0]
    step = diapason_len / points_num

    result = []
    for scale in scales:
        integrals = np.zeros(section_num)
        for i in range(points_num):
            spent = i / points_num
            t = diapason[0] + spent * diapason_len
            integrals[int(spent * section_num)] += func(t / scale) * step
        result.append(integrals)
    return np.stack(result)


# numpy realisation directly
def wt_str84wd(signal, scales, extend='zeros', wavelet='mexh'):
    signal_len = signal.shape[-1]
    func = mexh
    integrals = integrate(signal.shape[-1] - 1, scales, func=func)
    extended_signal = np.zeros(signal_len * 2)
    for i in range(signal_len * 2):
        if extend == 'zeros':
            if i < signal_len / 2 or i >= 3 / 2 * signal_len:
                extended_signal[i] = 0
            else:
                extended_signal[i] = signal[i - int(signal_len / 2)]
        else:
            extended_signal[i] = signal[(i - int(signal_len / 2)) % signal_len]

    result = np.zeros((len(scales), signal_len))

    for scale_num in range(len(scales)):
        for b in range(signal_len):
            current_corr = 0
            for i in range(signal_len - 1):
                current_corr += (extended_signal[b + i] + extended_signal[b + i + 1]) / 2 * integrals[scale_num][i]

            result[scale_num][b] = current_corr

    return result


def wt(signal, scales):
    integrals = integrate(signal.shape[-1], scales)

    result = []

    for scale_num in range(len(scales)):
        fft_wav = np.fft.fft(integrals[scale_num])
        fft_signal = np.fft.fft(signal)

        corr = np.fft.ifft(fft_wav.conj() * fft_signal)
        corr = np.roll(corr, -corr.shape[-1] // 2, axis=-1)  # spectrogram shift
        result.append(corr)

    return np.stack(result)
