# test_jittor.py
import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import jittor as jt
from jittor import nn
from utils.img_read_save import img_save
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import argparse
import time

# -------------------------
# 基础配置
# -------------------------
jt.flags.use_cuda = 1 if jt.has_cuda else 0

task_name = 'IVF'
dataset_name = 'MSRS'

# -------------------------
# 数据加载（替换成 Jittor Dataset）
# -------------------------
from utils.H5_read import H5ImageTextDataset  # 假设内部已兼容 Jittor
test_dataset = H5ImageTextDataset(os.path.join('VLFDataset_h5', dataset_name + '_test.h5'))

# Jittor DataLoader
from jittor.dataset import DataLoader
testloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

# -------------------------
# 模型加载
# -------------------------
from net.Film import Net
model = Net(hidden_dim=256, image2text_dim=32)
model.eval()

ckpt_path = os.path.join("models", task_name + '.pkl')
save_path = os.path.join("test_output", dataset_name, "Gray")
os.makedirs(save_path, exist_ok=True)

# 加载权重（Jittor 的方式）
ckpt = jt.load(ckpt_path)
model.load_parameters(ckpt["model"])

# -------------------------
# 推理
# -------------------------
with jt.no_grad():
    pbar = tqdm(total=len(testloader))
    for i, (data_IR, data_VIS, text, index) in enumerate(testloader):
        # 数据转成 Jittor 张量
        text = jt.array(text.squeeze(1), dtype=jt.float32)
        data_IR = jt.array(data_IR, dtype=jt.float32)
        data_VIS = jt.array(data_VIS, dtype=jt.float32)

        # 前向推理
        data_Fuse = model(data_IR, data_VIS, text)[0]

        # 归一化到 [0,1]
        data_Fuse = (data_Fuse - jt.min(data_Fuse)) / (jt.max(data_Fuse) - jt.min(data_Fuse))

        # 转 numpy
        fi = np.squeeze((data_Fuse * 255).numpy()).astype('uint8')

        # 保存
        img_save(fi, index[0], save_path)

        pbar.update(1)

