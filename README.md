# Explicitly Provable Gradient Network for Unrolled Medical Image Reconstruction Algorithms
*Baoshun Shi，Lijun Huang, Wenyuan Xu, Yueming Su*
## Abatract
Recently, unrolled algorithms have demonstrated outstanding empirical performance in solving medical image reconstruction problems; however, they face two primary limitations. First, proving that designed networks explicitly satisfy boundary or Lipschitz constraints—necessary for guaranteed convergence—is inherently challenging due to their black-box nature and limited interpretability. Second, the final imaging quality is limited because the prior network does not utilise the complementary information in intermediate images. We address these issues by proposing an explicitly provable tight frame–based gradient network and developing a convergent unrolled algorithm. Theoretically, we prove the Lipschitz constraint property of the gradient network. Furthermore, we generate reliable thresholds within the gradient network by introducing a threshold-generating sub-network. This sub-network explores complementary information from each intermediate image, including local frame coefficient details, non-local information and inter-stage information within the iterative algorithm. We propose a regularisation model based on the gradient network for solving medical image reconstruction problems and solve the corresponding optimisation problem using a convergent unrolled iterative algorithm. Extensive experimental results demonstrate the superiority of the proposed deep unrolled network in various medical image reconstruction tasks, including compressed sensing magnetic resonance imaging and sparse-view computed tomography reconstruction. Source code is made available at https://github.com/shibaoshun/GDUNet-csmri-svct.git.

![image name](https://github.com/shibaoshun/GDUNet/blob/main/TNN.png)

## Installation
The model is built in PyTorch 1.8.0 and  trained with NVIDIA 4090 GPU.
For installing, follow these intructions
```
conda create -n GDUNet python=3.7
conda activate GDUNet
pip install -r requirements.txt
```

## Dataset
You can download the testing datasets from the following Baidu Drive link:

📁 `data`

🔗 [https://pan.baidu.com/s/1oALqkIUqrEwDltJ7GXaNtg](https://pan.baidu.com/s/1oALqkIUqrEwDltJ7GXaNtg)

🔑 Password: `gbdb`

## Pre-trained Models  

📁 `result`

🔗 [https://pan.baidu.com/s/17R9Vlw05s1rvw32PaCVVWQ](https://pan.baidu.com/s/17R9Vlw05s1rvw32PaCVVWQ)

🔑 Password: `pfwj `

## CSMRI Result

![image-20250512164910204](https://github.com/shibaoshun/GDUNet/blob/main/fig3.png)

## SVCT reconstruction Result
![image-20250512164910204](https://github.com/shibaoshun/GDUNet/blob/main/fig7.jpg)
