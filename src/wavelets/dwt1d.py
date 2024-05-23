import colorsys

import numpy as np
import torch
from math import floor, ceil
from pywt import DiscreteContinuousWavelet, integrate_wavelet, scale2frequency
from pywt._cwt import next_fast_len
from pywt._extensions._pywt import _check_dtype, ContinuousWavelet, Wavelet

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
    # wavelet = DiscreteContinuousWavelet(wavelet)
    # precision = 12
    # int_psi, x = integrate_wavelet(wavelet, precision=precision)
    # int_psi_scales = []
    # for i, scale in enumerate(scales):
    #     step = x[1] - x[0]
    #     j = np.arange(scale * (x[-1] - x[0]) + 1) / (scale * step)
    #     j = j.astype(int)  # floor
    #     if j[-1] >= int_psi.size:
    #         j = np.extract(j < int_psi.size, j)
    #     int_psi_scale = int_psi[j][::-1]
    #     int_psi_scales.append(torch.tensor(int_psi_scale.copy()).to(device))
    # return int_psi_scales

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

# def get_representation(data):
#     if data.dtype != torch.complex64:
#         print('Data is not complex!')
#         return
#
#     input_shape = data.shape # shape needs to be (n_batch, 22, 80, 200)
#     data = data.view(-1, input_shape[2], input_shape[3])
#     amp = torch.abs(data)
#     phase = torch.angle(data)
#     H = phase / (2 * torch.pi)
#     L = 1 - 2 ** (-amp)
#     S = torch.ones_like(amp)
#
#     repr = torch.stack([H, L, S]) # now (3, -1, 80, 200)
#     repr = repr.permute(1, 0, 2, 3) # do (-1, 3, 80, 200)
#     repr_np = repr.numpy()
#
#     repr_np_rgb = np.zeros_like(repr_np)
#     for n in range(repr_np.shape[0]):
#         for y in range(repr_np.shape[2]):
#             for x in range(repr_np.shape[3]):
#                 h = repr_np[n, 0, y, x]
#                 l = repr_np[n, 1, y, x]
#                 s = repr_np[n, 2, y, x]
#                 r, g, b = colorsys.hls_to_rgb(h, l, s)
#                 repr_np_rgb[n, 0, y, x] = r
#                 repr_np_rgb[n, 1, y, x] = g
#                 repr_np_rgb[n, 2, y, x] = b
#
#     return torch.tensor(repr_np_rgb)
