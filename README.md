# 视觉语言大模型引导的多模态图像融合
复现的代码和数据集来源于论文 ***Image Fusion via Vision-Language Model (ICML 2024).***

## 🏊链接
- [*[模型权重]*](https://pan.baidu.com/s/1CT7I4YrhhgCUnuInaau05w?pwd=q45e)  
- [*[数据集（使用前200对）]*](https://pan.baidu.com/s/1acy8qxiDxSXChMisoh8sgQ?pwd=r3rs)  

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
您将得到训练的H5文件存放在目录 ``'./VLFDataset_h5/MSRS_train.h5'``.

其中，每个H5文件都包含三个字段imageA、imageB、text
```
text.shape：(1, 149, 768)
imageA.shape：(1, 288, 384)
imageB.shape：(1, 288, 384)
``` 

**4. FILM Training**

Run 
```
python train.py
``` 

The training results will be stored in the ``'./exp/'`` folder, with subfolder names that can be modified using the ``save_path`` variable in  ``'train.py'``. The next-level subfolders are named after the training start time and contain three folders: ``'code'``, ``'model'``, and ``'pic_fusion'``, as well as a log file and a JSON file recording the parameters. The ``'code'`` folder saves the model and training files for that session, the ``'model'`` folder saves the model weights for each epoch during training, and the ``'pic_fusion'`` folder saves the original images and fusion results of the first two batches from each training epoch.

### 🏄 Testing

**1. Pretrained models**

The pre-trained models can be found at ``'./models/IVF.pth'``, ``'./models/MEF.pth'``, ``'./models/MFF.pth'``, and ``'./models/MIF.pth'``. These models are responsible for infrared-visible fusion (IVF), multi-exposure image fusion (MEF), multi-focus image fusion (MFF), and medical image fusion (MIF) tasks, respectively.

**2. Test datasets**

The test datasets used in the paper are provided in the format ``'./VLFDataset/{Task_name}/{Dataset_name}/test.txt'``. Here, the provided ``'Task_name'`` includes IVF, MEF, MFF, and MIF, and ``'Dataset_name'`` corresponds to the dataset names included for each task.

Unfortunately, due to the size of the datasets provided in the paper exceeding 4GB, we are unable to upload them for exhibition. You can download them via [this link](https://drive.google.com/drive/folders/1JPNbh-iFhkbr35FDUOYEN4WE4LnxY954?usp=sharing). This includes images, text, and the implicit features corresponding to the text (via BLIP2) for all datasets.

**3. Results in Our Paper**

If you want to infer with our FILM and obtain the fusion results in our paper, please run 
```
python test.py
``` 
to perform image fusion. The output fusion results will be saved in the ``'./test_output/{Dataset_name}/Gray'``  folder. If you want to test using your own trained model, you can set the path of the model you want to load as ``'ckpt_path'`` before the model weights are loaded in ``'test.py'``.

The output for IVF is:

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
which can match the results in Table 1 in our original paper.

The output for MIF is:

```
================================================================================
The test result of Harvard:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            4.74	65.23	23.36	6.19	0.78	0.76    
================================================================================
```
which can match the results in Table 2 in our original paper.

The output for MEF is:

```
================================================================================
The test result of SICE:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.07	54.22	19.39	5.14	1.05	0.79    
================================================================================

================================================================================
The test result of MEFB:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.32	68.98	20.94	6.14	0.98	0.77    
================================================================================
```
which can match the results in Table 3 in our original paper.

The output for MFF is:

```
================================================================================
The test result of RealMFF:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.11	54.97	15.62	5.43	1.11	0.76   
================================================================================

================================================================================
The test result of Lytro:
                 EN      SD      SF      AG     VIFF     Qabf    
FILM            7.56	59.15	19.57	6.97	0.98	0.74    
================================================================================
```
which can match the results in Table 4 in our original paper.

## 📁 Vision-Language Fusion (VLF) Dataset
Considering the high computational cost of invoking various vision-language components, and to facilitate subsequent research on image fusion based on vision-language models, we propose the **VLF Dataset**. This dataset encompasses paired paragraph descriptions generated by ChatGPT, covering all image pairs from the training and test sets of the eight widely-used fusion datasets.

These include

* **Infrared-visible image fusion (IVF)**: [MSRS](https://github.com/Linfeng-Tang/MSRS), [M³FD](https://github.com/JinyuanLiu-CV/TarDAL), and [RoadScene](https://github.com/hanna-xu/RoadScene) datasets;
* **Medical image fusion (MIF)**: [Harvard](http://www.med.harvard.edu/AANLIB/home.html) dataset;
* **Multi-exposure image fusion (MEF)**: [SICE](https://github.com/csjcai/SICE) and [MEFB](https://github.com/xingchenzhang/MEFB) datasets;
* **Multi-focus image fusion (MFF)**: [RealMFF](https://github.com/Zancelot/Real-MFF) and [Lytro](http://mansournejati.ece.iut.ac.ir/content/lytro-multi-focus-dataset) datasets;

**The dataset is available for download via [Google Drive](https://drive.google.com/drive/folders/1JPNbh-iFhkbr35FDUOYEN4WE4LnxY954?usp=sharing).**

**More visualizations and illustrations of the VLF Dataset can be found on our [Project Homepage](https://zhaozixiang1228.github.io/Project/IF-FILM/).**

<u>**[Notice]**</u>:  
Considering the immense workload involved in creating this dataset, we have opened a [Google Form](https://forms.gle/1kTAS15DktRPLs5BA) for error correction feedback. Please provide your suggestions for correcting any errors in the VLF dataset. If you have any questions regarding the Google Form, please contact [Zixiang](https://zhaozixiang1228.github.io/) via email.


## 🙌 Fusion via vIsion-Language Model (FILM)

#### Workflow for our FILM

<img src="images\1.png" width="70%" align=center />

#### Detailed Architecture of FILM

<img src="images\2.png" width="70%" align=center />

#### Visualization of the VLF dataset

<img src="images\3.png" width="90%" align=center />

## 📝 Experimental Results

### Infrared-visible image fusion (IVF)

#### Qualitative fusion results:
<img src="images\RS_06570.png" width="70%" align=center />

#### Quantitative fusion results:
<img src="images\R1.png" width="90%" align=center />

### Medical image fusion (MIF)

#### Qualitative fusion results:
<img src="images\Harvard_MRI_PET_19.png" width="50%" align=center />

#### Quantitative fusion results:
<img src="images\R2.png" width="40%" align=center />

### Multi-exposure image fusion (MEF)

#### Qualitative fusion results:
<img src="images\SICE_076.png" width="70%" align=center />

#### Quantitative fusion results:
<img src="images\R3.png" width="70%" align=center />

### Multi-focus image fusion (MFF)

#### Qualitative fusion results:
<img src="images\RealMFF_126.png" width="70%" align=center />

#### Quantitative fusion results:
<img src="images\R4.png" width="70%" align=center />

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Zhaozixiang1228/IF-FILM&type=Date)](https://star-history.com/#Zhaozixiang1228/IF-FILM&Date)

## 📖 Related Work

- [*Equivariant Multi-Modality Image Fusion.*](https://arxiv.org/abs/2305.11443) **CVPR 2024**.    
Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, Luc Van Gool.

- [*DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion.* ](https://arxiv.org/abs/2303.06840) **ICCV 2023 (Oral)**.    
Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, Luc Van Gool. 

- [*CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion.*](https://arxiv.org/abs/2211.14461) **CVPR 2023**.    
Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, Luc Van Gool.

- [*DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion.*](https://arxiv.org/abs/2305.11443) **IJCAI 2020**.    
Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang and Pengfei Li.

- [*Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling.*](https://ieeexplore.ieee.org/document/9416456.) **IEEE TCSVT 2021**.    
Zixiang Zhao, Shuang Xu, Jiangshe Zhang, Chengyang Liang, Chunxia Zhang and Junmin Liu.

<!-- - Zixiang Zhao, Jiangshe Zhang, Haowen Bai, Yicheng Wang, Yukun Cui, Lilun Deng, Kai Sun, Chunxia Zhang, Junmin Liu, Shuang Xu. *Deep Convolutional Sparse Coding Networks for Interpretable Image Fusion.* **CVPR Workshop 2023**. https://robustart.github.io/long_paper/26.pdf.

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang. *Bayesian fusion for infrared and visible images.* **Signal Processing**. https://doi.org/10.1016/j.sigpro.2020.107734. -->

