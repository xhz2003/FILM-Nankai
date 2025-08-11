import warnings
import jittor as jt
from jittor import nn
import jittor.nn as F


def _fspecial_gauss_1d(size, sigma):
    """生成一维高斯核"""
    coords = jt.arange(size, dtype=jt.float32)
    coords -= size // 2
    g = jt.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    """使用一维高斯核平滑输入"""
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = nn.conv2d
    elif len(input.shape) == 5:
        conv = nn.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            perms = list(range(win.ndim))
            perms[2 + i], perms[-1] = perms[-1], perms[2 + i]
            out = conv(out, weight=win.permute(perms), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    return out

def _ssim(X, Y, data_range, win, K=(0.01, 0.03)):
    K1, K2 = K
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win.astype(X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = ssim_map.reshape((ssim_map.shape[0], ssim_map.shape[1], -1)).mean(-1)
    cs = cs_map.reshape((cs_map.shape[0], cs_map.shape[1], -1)).mean(-1)
    return ssim_per_channel, cs

def ssim(X, Y, data_range=255, size_average=True,
         win_size=11, win_sigma=1.5, win=None,
         K=(0.01, 0.03), nonnegative_ssim=False):

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if win is not None:
        win_size = win.shape[-1]
    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, _ = _ssim(X, Y, data_range=data_range, win=win, K=K)
    if nonnegative_ssim:
        ssim_per_channel = F.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(dim=1)

def ms_ssim(X, Y, data_range=255, size_average=True,
            win_size=11, win_sigma=1.5, win=None,
            weights=None, K=(0.01, 0.03)):

    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")
    if not X.dtype == Y.dtype:
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.pool2d
    elif len(X.shape) == 5:
        avg_pool = F.pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:
        win_size = win.shape[-1]
    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** 4), \
        "Image size should be larger due to the 4 downsamplings in ms-ssim"

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    weights = jt.array(weights, dtype=X.dtype)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, K=K)

        if i < levels - 1:
            mcs.append(F.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, stride=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, stride=2, padding=padding)

    ssim_per_channel = F.relu(ssim_per_channel)
    mcs_and_ssim = jt.stack(mcs + [ssim_per_channel], dim=0)
    ms_ssim_val = jt.prod(mcs_and_ssim ** weights.reshape((-1, 1, 1)), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(dim=1)

class SSIM(nn.Module):
    def __init__(self, data_range=255, size_average=True, win_size=11,
                 win_sigma=1.5, channel=3, spatial_dims=2,
                 K=(0.01, 0.03), nonnegative_ssim=False):
        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def execute(self, X, Y):
        return ssim(X, Y, data_range=self.data_range,
                    size_average=self.size_average, win=self.win,
                    K=self.K, nonnegative_ssim=self.nonnegative_ssim)

class MS_SSIM(nn.Module):
    def __init__(self, data_range=255, size_average=True, win_size=11,
                 win_sigma=1.5, channel=3, spatial_dims=2,
                 weights=None, K=(0.01, 0.03)):
        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def execute(self, X, Y):
        return ms_ssim(X, Y, data_range=self.data_range,
                       size_average=self.size_average, win=self.win,
                       weights=self.weights, K=self.K)
