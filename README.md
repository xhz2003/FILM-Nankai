# 视觉语言大模型引导的多模态图像融合
复现的代码和数据集来源于论文 ***Image Fusion via Vision-Language Model (ICML 2024).***

## 🏊链接
- [*[模型权重]*](https://pan.baidu.com/s/1CT7I4YrhhgCUnuInaau05w?pwd=q45e)  
- [*[数据集（使用前1500对）]*](https://pan.baidu.com/s/1acy8qxiDxSXChMisoh8sgQ?pwd=r3rs)  

## 🌐Usage
### 训练
**1. 虚拟环境配置**
```
conda create -n FILM python=3.8
conda activate FILM
#注意jittor库安装
sudo apt install python3.8-dev libomp-dev
python3.8 -m pip install jittor
python3.8 -m jittor.test.test_example
pip install -r requirements.txt
```

**2. 数据准备与预处理**

下载好上述链接中的数据集后，按照下载的层级关系放在VLFDataset文件夹中，然后运行下面的命令
Run 
```
python data_process.py
``` 
您将得到训练的H5文件存放在目录 ``'./VLFDataset_h5/MSRS_train.h5'``。其中，每个H5文件都包含三个字段imageA、imageB、text
```
text.shape：(1, 149, 768)
imageA.shape：(1, 288, 384)
imageB.shape：(1, 288, 384)
``` 

**4. FILM 训练**

Run 
```
python train.py
``` 
训练结果将存储在 './exp/' 文件夹中。下一级子文件夹内部包含三个文件夹（'code'、'model' 和 'pic_fusion'），以及一个日志文件和一个记录参数的 JSON 文件。其中：

'code' 文件夹用于保存该次训练对应的模型文件和训练文件；

'model' 文件夹用于保存训练过程中每个 epoch 的模型权重；

'pic_fusion' 文件夹用于保存每个训练 epoch 中前两个批次的原始图像及融合结果。

在模型保存方面，jittor和pytorch有所区别，因此进行改动
```
jt.save(checkpoint, os.path.join(model_dir, f'ckpt_{epoch+1}.pkl')) #保存
model.load(jt.load(ckpt_path)["model"]) #加载
``` 
即保存的是一个字典，键是参数名，值是 Jittor 的 jt.Var对象。
### 测试

**1. 预训练模型**

预训练模型可在- [*[链接]*](https://pan.baidu.com/s/1CT7I4YrhhgCUnuInaau05w?pwd=q45e)中找到。用于处理红外-可见光融合（IVF）任务，其他的多曝光图像融合、多聚焦图像融合、和医学图像融合等任务也是一样的测试代码。

**2. 测试数据集**

通过上面提供的数据集下载链接- [*[数据集（使用前1500对）]*](https://pan.baidu.com/s/1acy8qxiDxSXChMisoh8sgQ?pwd=r3rs)是包括测试集的，需要在data_process.py文件中修改：
```
img_text_path = 'VLFDataset'
h5_path = "VLFDataset_h5"
os.makedirs(h5_path, exist_ok=True)
task_name = 'IVF'
dataset_name = 'MSRS'
dataset_mode = 'test'
size = 'small'
```
一共是361对图片-文本数据集，722张

<img src="images\测试数据集处理.png" width="70%" align=center />

**3. 实验结果对比**

 （1）使用FILM 模型进行推理并得到融合结果，请运行以下命令进行图像融合：
```
python test.py
```
输出的融合结果将保存在 './test_output/{Dataset_name}/Gray' 文件夹中。同时可以在 'test.py' 中加载模型权重之前，将您要加载的模型路径设置为 'ckpt_path'。
融合结果如下所示（这里有疑惑的是同样的数据集和配置，jittor在推理时速度要慢很多，可能是没有设置GPU或者未加载成功所导致的）：

<img src="images\融合结果.jpg" width="70%" align=center />

（2）融合结果对比

使用jittor框架在小数据集（150对）上面训练了100个epoch，训练日志[*[链接]*](./log)，训练的损失函数曲线如下：

<img src="images\loss_comparison.png" width="70%" align=center />

分析

**数值精度与计算细节。** Jittor在底层计算和数值稳定性上可能有不同的实现，尤其在反向传播和浮点运算精度上会有差异，可能导致训练时梯度波动更明显。

**反向传播的计算路径。** PyTorch采用静态图与动态图结合的混合计算模式，而 Jittor 是纯动态图架构。

（3）定量分析

```
================================================================================
The test result of MSRS:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            6.72	43.64	11.55	3.72	1.06	0.70    
================================================================================

================================================================================
The test result of RoadScene:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.43	49.26	17.33	6.59	0.69	0.62    
================================================================================

================================================================================
The test result of M3FD:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.09	41.52	16.77	5.55	0.83	0.67    
================================================================================
```
