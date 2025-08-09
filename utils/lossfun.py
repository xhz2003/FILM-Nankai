import jittor as jt
import jittor.nn as F
import numpy as np
from jittor import nn
# -----------------------------
# 工具函数：高斯窗口
# -----------------------------
def gaussian(window_size, sigma):
    x = jt.array([-(window_size // 2) + i for i in range(window_size)])
    gauss = jt.exp(-(x ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # [ws, 1]
    _2D_window = jt.matmul(_1D_window, _1D_window.t())  # [ws, ws]
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
    window = _2D_window.broadcast([channel, 1, window_size, window_size])
    return window

# -----------------------------
# BCE Loss
# -----------------------------
class BCE_loss:
    def __init__(self):
        pass

    def cal(self, predictlabel, truelabel):
        sum_truelabel = jt.sum(truelabel, dim=2)
        validindex = jt.where(sum_truelabel == 1)
        # 提取有效位置的数据
        predict_valid = predictlabel[validindex]
        target_valid = truelabel[validindex]

        # 使用 BCELoss
        criteria = nn.BCELoss()
        loss = criteria(predict_valid, target_valid)
        return loss

# -----------------------------
# MSE Loss (基于 argmax 归一化索引)
# -----------------------------
class MSE_loss:
    def __init__(self):
        pass

    def cal(self, predictlabel, truelabel):
        sum_truelabel = jt.sum(truelabel, dim=2)
        validindex = jt.where(sum_truelabel == 1)

        predict_valid = predictlabel[validindex]
        target_valid = truelabel[validindex]

        # 只取 argmax 的 indices  否则会出现报错
        _, label_indices = jt.argmax(target_valid, dim=1)
        _, pred_indices = jt.argmax(predict_valid, dim=1)

        label_index = label_indices.float() / target_valid.shape[1]
        pred_index = pred_indices.float() / target_valid.shape[1]

        criteria = nn.MSELoss()
        loss = criteria(label_index, pred_index)
        return loss


# -----------------------------
# BCE + MSE 混合损失
# -----------------------------
class BCE_MSE_loss:
    def __init__(self, balan_para):
        self.bp = balan_para
        self.BCEloss = BCE_loss()
        self.MSEloss = MSE_loss()

    def cal(self, predictlabel, truelabel):
        bce_loss = self.BCEloss.cal(predictlabel, truelabel)
        mse_loss = self.MSEloss.cal(predictlabel, truelabel)
        loss = bce_loss + self.bp * mse_loss
        return loss


# -----------------------------
# 交叉熵损失 (CE_Loss)
# -----------------------------
def CE_Loss(inputs, target, num_classes=0):
    n, c, h, w = inputs.shape
    nt, ht, wt = target.shape

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # reshape: [N,C,H,W] -> [N*H*W, C]
    temp_inputs = inputs.permute(0, 2, 3, 1).view(-1, c)
    temp_target = target.view(-1)

    criteria = nn.CrossEntropyLoss(ignore_index=num_classes)
    ce_loss = criteria(temp_inputs, temp_target)
    return ce_loss


# -----------------------------
# Sobel 梯度提取模块
# -----------------------------
class Sobelxy(nn.Module):
    def __init__(self, device='cuda'):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        # 直接赋值为成员变量，不需要 Parameter
        self.weightx = jt.array(kernelx).float().unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
        self.weighty = jt.array(kernely).float().unsqueeze(0).unsqueeze(0)

    def execute(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return jt.abs(sobelx) + jt.abs(sobely)


# -----------------------------
# Fusion Loss
# -----------------------------
class Fusionloss(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=10, in_max=True, device='cuda'):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy(device=device)
        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad
        self.in_max = in_max

    def execute(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]  # 取第一个通道

        if self.in_max:
            x_in_max = jt.maximum(image_y, image_ir)
        else:
            x_in_max = (image_y + image_ir) / 2.0

        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        gen_grad = self.sobelconv(generate_img)

        x_grad_joint = jt.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, gen_grad)

        loss_total = self.coeff_int * loss_in + self.coeff_grad * loss_grad
        return loss_total, loss_in, loss_grad


# -----------------------------
# MEF-SSIM 核心函数
# -----------------------------
def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    # 获取输入张量的形状
    K, C, H, W = Ys.shape

    # 计算参考图像Ys的统计量
    muY_seq = nn.conv2d(Ys, window, padding=(ws // 2, ws // 2), groups=C)
    muY_seq = muY_seq.view(K, C, H, W)
    muY_sq_seq = muY_seq * muY_seq

    # 计算sigmaY_sq_seq
    conv_result = nn.conv2d(Ys * Ys, window, padding=(ws // 2, ws // 2), groups=C)
    sigmaY_sq_seq = conv_result - muY_sq_seq

    # 计算最大值和索引
    sigmaY_sq = jt.max(sigmaY_sq_seq, dim=0)
    if isinstance(sigmaY_sq, tuple):
        sigmaY_sq = sigmaY_sq[0]

    # 处理argmax返回值
    patch_index = jt.argmax(sigmaY_sq_seq, dim=0)
    if isinstance(patch_index, tuple):
        patch_index = patch_index[0]

    # 计算测试图像X的统计量
    muX = nn.conv2d(X, window, padding=(ws // 2, ws // 2), groups=C)
    muX = muX.view(C, H, W)
    muX_sq = muX * muX

    conv_x_result = nn.conv2d(X * X, window, padding=(ws // 2, ws // 2), groups=C)
    sigmaX_sq = conv_x_result - muX_sq
    if isinstance(sigmaX_sq, tuple):
        sigmaX_sq = sigmaX_sq[0]

    # 计算相关项
    broadcasted_X = jt.broadcast(X, Ys)
    if isinstance(broadcasted_X, tuple):
        broadcasted_X = broadcasted_X[0]

    xy_product = broadcasted_X * Ys
    conv_xy_result = nn.conv2d(xy_product, window, padding=(ws // 2, ws // 2), groups=C)

    broadcasted_muX = jt.broadcast(muX, muY_seq)
    if isinstance(broadcasted_muX, tuple):
        broadcasted_muX = broadcasted_muX[0]

    sigmaXY = conv_xy_result - broadcasted_muX * muY_seq
    sigmaXY = sigmaXY.view(K, C, H, W)
    if isinstance(sigmaXY, tuple):
        sigmaXY = sigmaXY[0]

    # 计算质量图
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    if isinstance(cs_seq, tuple):
        cs_seq = cs_seq[0]

    # 处理gather操作
    cs_seq_reshaped = cs_seq.view(K, -1)
    patch_index_reshaped = patch_index.view(1, -1)

    gathered = jt.gather(cs_seq_reshaped, 0, patch_index_reshaped)
    if isinstance(gathered, tuple):
        gathered = gathered[0]

    cs_map = gathered.view(C, H, W)

    if is_lum:
        muY_seq_reshaped = muY_seq.view(K, -1)
        lY = jt.mean(muY_seq_reshaped, dim=1)
        if isinstance(lY, tuple):
            lY = lY[0]

        lL = jt.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        if isinstance(lL, tuple):
            lL = lL[0]

        # 修复：使用jt.broadcast替代broadcast_to方法
        lG_base = jt.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None]
        lG = jt.broadcast(lG_base, lL)
        if isinstance(lG, tuple):
            lG = lG[0]

        LY = lG * lL
        muY = jt.sum((LY * muY_seq), dim=0) / jt.sum(LY, dim=0)
        if isinstance(muY, tuple):
            muY = muY[0]

        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
        if isinstance(l_map, tuple):
            l_map = l_map[0]
    else:
        l_map = jt.array([1.0])
    if full:
        l = jt.mean(l_map)
        if isinstance(l, tuple):
            l = l[0]

        cs = jt.mean(cs_map)
        if isinstance(cs, tuple):
            cs = cs[0]

        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()
    if isinstance(q, tuple):
        q = q[0]

    return q


# test_loss_jittor.py
class MEFSSIM(nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, channel)
        self.denom_g = 2 * (sigma_g ** 2)
        self.denom_l = 2 * (sigma_l ** 2)
        self.C1 = c1 ** 2
        self.C2 = c2 ** 2
        self.is_lum = is_lum

    def execute(self, X, Ys):
        (_, channel, _, _) = Ys.shape
        recreate = False
        if channel != self.channel:
            recreate = True
        try:
            if self.window.dtype != X.dtype:
                recreate = True
        except Exception:
            pass

        if recreate:
            window = create_window(self.window_size, channel, dtype=X.dtype, device='cuda' if X.is_cuda else None)
            self.window = window
            self.channel = channel
        else:
            window = self.window

        return _mef_ssim(X, Ys, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)

# -----------------------------
# LpLssimLossweight Jittor 版（模块）
# -----------------------------
def ssim_map(img1, img2, window, window_size, channel, size_average=True):
    """
    img1, img2: [N, C, H, W]
    window: [C,1,ws,ws] (grouped conv)
    returns scalar mean SSIM (or per-sample mean if size_average=False)
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class LpLssimLossweight(nn.Module):
    def __init__(self, window_size=5, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        return gaussian(window_size, sigma)

    def create_window(self, window_size, channel):
        return create_window(window_size, channel)

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        return ssim_map(img1, img2, window, window_size, channel, size_average)

    def execute(self, image_in, image_out, weight):
        (_, channel, _, _) = image_in.shape
        recreate = False
        if channel != self.channel:
            recreate = True
        try:
            if self.window.dtype != image_in.dtype:
                recreate = True
        except Exception:
            pass

        if recreate:
            window = self.create_window(self.window_size, channel)
            window = window.cuda()
            try:
                window = window.astype(image_in.dtype)
            except Exception:
                window = window.float()
            self.window = window
            self.channel = channel
        else:
            window = self.window

        diff = image_in - image_out
        Lp = jt.sqrt(jt.sum(diff * diff) + 1e-12)
        Lssim = 1.0 - self._ssim(image_in, image_out, window, self.window_size, self.channel, self.size_average)

        return Lp + Lssim * weight, Lp, Lssim * weight