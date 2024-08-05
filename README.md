# MRBFN-SPSD
The conventional radial basis function network is modified for joint feature selection and classification. By utilizing the anisotropic Gaussian basis function, an individual weight parameter is assigned to each feature for feature weighting. A safe dynamic sparse training method is proposed to optimize all the network parameters, which can gradually identify and
zero out the redundant feature and output weights during training. The resulting model sparsity and optimized feature weights help to improve generalization and interpretability of the constructed model.

If you use the matlab code here, please cite our paper below:
Xusheng Qian, Jisu Hu, Yi Zheng, He Huang, Zhiyong Zhou, Yakang Dai*. Safe dynamic sparse training of modified RBF networks for joint feature selection and classification. Neurocomputing, 2024, 600: 128150. 
https://www.sciencedirect.com/science/article/pii/S0925231224009214

ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Yakang Dai (daiyk@sibet.ac.cn) or Xusheng Qian (qianxs@sibet.ac.cn).

Requirement: The package was developed with MATLAB 2019a.

Usage：
run main.m
