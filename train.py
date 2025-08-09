import os
import matplotlib.pyplot as plt
import sys
import time
import datetime
import numpy as np
import shutil
import re
import argparse
import jittor as jt
from jittor import nn, optim, lr_scheduler
from jittor.dataset import Dataset
import h5py

# 日志和可视化
from utils.Logger import Logger1
from tensorboardX import SummaryWriter

# 损失函数（需适配 Jittor）
from utils.lossfun import Fusionloss, LpLssimLossweight

# 模型
from net.Film import Net
from utils.H5_read import H5ImageTextDataset
# 忽略警告
import warnings

warnings.filterwarnings('ignore')

# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
jt.flags.use_cuda = 1  # 启用 CUDA

sys.path.append(os.getcwd())

# 解析参数（保持不变）
parser = argparse.ArgumentParser()
parser.add_argument('--i2t_dim', type=int, default=32, help='')
parser.add_argument('--hidden_dim', type=int, default=256, help='')
parser.add_argument('--numepochs', type=int, default=150, help='')
parser.add_argument('--lr', type=float, default=1e-5, help='')
parser.add_argument('--gamma', type=float, default=0.6, help='')
parser.add_argument('--step_size', type=int, default=50, help='')
parser.add_argument('--batch_size', type=int, default=2, help='')
parser.add_argument('--loss_grad_weight', type=int, default=20, help='')
parser.add_argument('--loss_ssim', type=int, default=0, help='')
parser.add_argument('--dataset_path', type=str, default="VLFDataset_h5/MSRS_train.h5", help='')
opt = parser.parse_args()

'''
------------------------------------------------------------------------------
Set the hyper-parameters for training
------------------------------------------------------------------------------
'''
pre_model = ""
num_epochs = opt.numepochs
lr = opt.lr
step_size = opt.step_size
gamma = opt.gamma
weight_decay = 0
batch_size = opt.batch_size
weight_ingrad = opt.loss_grad_weight
weight_ssim = opt.loss_ssim
hidden_dim = opt.hidden_dim
i2t_dim = opt.i2t_dim
dataset_path = opt.dataset_path
exp_name = ''
'''
------------------------------------------------------------------------------
Model & Loss
------------------------------------------------------------------------------
'''
#无需DataParallel
model = Net(hidden_dim=hidden_dim, image2text_dim=i2t_dim)
if pre_model != "":
    ckpt = jt.load(pre_model)
    model.load_state_dict(ckpt['model'])
    print('load_pretrain_model')

criterion = LpLssimLossweight()
loss_fn = Fusionloss(coeff_grad=weight_ingrad)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# 数据加载器----改变----- jittor与torch的直接Dataloader不一样
train_dataset = H5ImageTextDataset(dataset_path)
train_dataset.set_attrs(batch_size=batch_size, shuffle=True, drop_last=True)

'''
------------------------------------------------------------------------------
Logging & Checkpoint
------------------------------------------------------------------------------
'''
time_begin = time.strftime("%y_%m_%d_%H_%M", time.localtime())
save_path = "exp/" + str(time_begin) + '_epochs_%s' % (
    str(opt.numepochs)) + '_lr_%s' % (str(opt.lr)) + '_stepsize_%s' % (str(opt.step_size)) + '_bs_%s' % (
                opt.batch_size) + '_gradweight_%s' % (str(opt.loss_grad_weight)) + '_gamma_%s' % (
                str(opt.gamma)) + exp_name

logger = Logger1(rootpath=save_path, timestamp=True)
params = {
    'epoch': num_epochs,
    'lr': lr,
    'batch_size': batch_size,
    'optim_step': step_size,
    'optim_gamma': gamma,
    'gradweight': weight_ingrad,
}
logger.save_param(params)
logger.new_subfolder('model')
writer = SummaryWriter(logger.logpath)
exp_folder = logger.get_timestamp_folder_name()
destination_folder = os.path.join(save_path, exp_folder, 'code')


def save_code_files(source_file, destination_folder):
    global model_file_path
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(source_file, 'r', encoding="utf-8") as file:
        content = file.read()
    match = re.search(r'from net\.(\w+) import Net', content)
    if match:
        model_name = match.group(1)
        model_file_path = os.path.join('net', f'{model_name}.py')
    dest_train_file_path = os.path.join(destination_folder, os.path.basename(__file__))

    shutil.copyfile(source_file, dest_train_file_path)
    if 'model_file_path' in globals():
        shutil.copyfile(model_file_path, os.path.join(destination_folder, f'{model_name}.py'))


save_code_files(__file__, destination_folder)

'''
------------------------------------------------------------------------------
Training Loop
------------------------------------------------------------------------------
'''
step = 0
start_time = time.time()

for epoch in range(num_epochs):
    s_temp = time.time()
    model.train()
    for i, (data_IR, data_VIS, text, index) in enumerate(train_dataset):
        text = text.squeeze(1)  # 去掉第1维 → (2, 149, 768)
        F = model(data_IR, data_VIS, text)

        batchsize, channels, rows, columns = data_IR.shape
        weighttemp = int(np.sqrt(rows * columns))

        lplssimA, lpA, lssimA = criterion(image_in=data_IR, image_out=F, weight=weighttemp)
        lplssimB, lpB, lssimB = criterion(image_in=data_VIS, image_out=F, weight=weighttemp)
        loss_in_grad, _, _ = loss_fn(data_IR, data_VIS, F)
        loss_ssim = lplssimA + lplssimB
        lossALL = loss_in_grad + weight_ssim * loss_ssim

        optimizer.step(lossALL)

        batches_done = epoch * len(train_dataset) + i
        batches_left = num_epochs * len(train_dataset) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - time.time()))  # 近似
        prev_time = time.time()

        logger.log_and_print(
            "[Epoch %d/%d] [Batch %d/%d] [loss: %f] [loss_in_grad: %f] [lplssimA: %f] [lplssimB: %f] ETA: %s"
            % (
                epoch + 1,
                num_epochs,
                i,
                len(train_dataset),
                lossALL.item(),
                loss_in_grad.item(),
                lplssimA.item(),
                lplssimB.item(),
                str(time_left).split('.')[0]
            )
        )

        # TensorBoard 记录
        writer.add_scalar('loss/01 Loss', lossALL.item(), step)
        writer.add_scalar('loss/01 loss_in_grad', loss_in_grad.item(), step)
        writer.add_scalar('loss/01 lplssimA', lplssimA.item(), step)
        writer.add_scalar('loss/01 lplssimB', lplssimB.item(), step)
        writer.add_scalar('loss/14 learning rate', optimizer.lr, step)
        step += 1

        # 每 epoch 保存图片
        if (epoch + 1) % 1 == 0 and i < 2:
            for j in range(data_IR.shape[0]):
                temp = np.zeros((rows, 3 * columns))
                temp[:rows, 0:columns] = data_IR[j].numpy().squeeze() * 255
                temp[:rows, columns:2 * columns] = data_VIS[j].numpy().squeeze() * 255
                temp[:rows, 2 * columns:3 * columns] = F[j].numpy().squeeze() * 255

                pic_dir = os.path.join(logger.logpath, 'pic_fusion', f"ckpt_{epoch + 1}")
                os.makedirs(pic_dir, exist_ok=True)
                plt.imsave(os.path.join(pic_dir, f"{index[j]}.png"), temp, cmap="gray")

    scheduler.step()

    # 保存 checkpoint
    if (epoch + 1) % 1 == 0:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer1': optimizer.state_dict(),
            'lr_schedule1': {
                'last_epoch': scheduler.last_epoch,
                'step_size': scheduler.step_size,
                'gamma': scheduler.gamma,
            },
            "epoch": epoch,
            'step': step,
        }
        model_dir = os.path.join(logger.logpath, 'model')
        os.makedirs(model_dir, exist_ok=True)
        #模型保存
        jt.save(checkpoint, os.path.join(model_dir, f'ckpt_{epoch + 1}.pkl'))

        e_temp = time.time()
        print("This Epoch takes time: " + str(e_temp - s_temp))

end_time = time.time()
logger.log_and_print("total_time: " + str(end_time - start_time))

# 关闭 writer
writer.close()
