# [Industrial Image Anomaly Localization Based on Gaussian Clustering of Pre-trained Feature](https://ieeexplore.ieee.org/document/9479740)

## Testing

```python
python dpfc_kmeans_covariance.py --k 10 --backbone wide_resnet50_2 --data_root mvtec_root
```

## Abstract

Anomaly localization is valuable for improvement of complex production processing in smart manufacturing system. As the distribution of anomalies is unknowable and labeled data is few, unsupervised methods based on convolutional neural network (CNN) have been studied for anomaly localization. But there are still problems for real industrial applications, in terms of localization accuracy, computation time, and memory storage. This article proposes a novel framework called as Gaussian clustering of pretrained feature (GCPF), including the clustering and inference stage, for anomaly localization in unsupervised way. The GCPF consists of three modules which include pretrained deep feature extraction (PDFE), multiple independent multivariate Gaussian clustering (MIMGC), and multihierarchical anomaly scoring (MHAS). In the clustering stage, features of normal images are extracted by pretrained CNN at the PDFE module, and then clustered at the MIMGC module. In the inference stage, features of target images are extracted and then scored for anomaly localization at the MHAS module. The GCPF is compared with the state-of-the-art methods on MVTec dataset, achieving receiver operating characteristic curve of 96.86% over all 15 categories, and extended to NanoTWICE and DAGM datasets. The GCPF outperforms the compared methods for unsupervised anomaly localization, and significantly reserves the low computation complexity and online memory storage which are important for real industrial applications.

![GCPF](./gcpf.bmp)

![Results](./results.bmp)

## Citation

```
@ARTICLE{9479740,
author={Wan, Qian and Gao, Liang and Li, Xinyu and Wen, Long},
 journal={IEEE Transactions on Industrial Electronics},
title={Industrial Image Anomaly Localization Based on Gaussian Clustering of Pretrained Feature},
year={2022},
 volume={69},
number={6},
pages={6182-6192},
doi={10.1109/TIE.2021.3094452}
}
```


