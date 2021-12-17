# deep pretrained feature clustering for unsupervised anomaly detection

import os
import argparse
from loguru import logger
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, wide_resnet50_2, wide_resnet101_2, resnet101, resnet152
from dataset.mvtec import MVTecDataset, MVTec_CLASS_NAMES
from torch.utils.data import DataLoader

import pickle
from utils import calculate_distance_matrix, visualize

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from embedding import _kmeans_fun_gpu, _kgaussians_fun_gpu

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser('DPFC-GMM')

    parser.add_argument("--backbone", type=str, default='wide_resnet50_2')

    parser.add_argument("--img_batch", type=int, default=32)
    parser.add_argument("--fea_batch", type=int, default=128)

    parser.add_argument("--data_root", type=str, default="D:/Dataset/mvtec_anomaly_detection/")
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--k", type=int, default=3)
    # parser.add_argument("--save_path", type=str, default='result_224')
    parser.add_argument("--save_path", type=str, default='result_imglevel')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_feas_path = os.path.join(args.save_path, 'dpfc_kmeans_covariance', "temp")
    os.makedirs(train_feas_path, exist_ok=True)
    args.save_path = os.path.join(args.save_path, 'dpfc_kmeans_covariance', f"{args.backbone}", f"k_{args.k}_{args.k}_{args.k}")
    os.makedirs(args.save_path, exist_ok=True)
    temp_path = os.path.join(args.save_path, "temp")
    os.makedirs(temp_path, exist_ok=True)

    # logging
    args.logger = args.save_path + f'/logger.txt'
    logger.add(args.logger, rotation="200 MB", backtrace=True, diagnose=True)
    logger.info(str(args))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    output_feas = []

    def _forward_hook(module, input, output):
        output_feas.append(output)

    # model = wide_resnet50_2(pretrained=True)
    model = eval(args.backbone)(pretrained=True)
    # model.layer1[-1].register_forward_hook(_forward_hook)
    # model.layer2[-1].register_forward_hook(_forward_hook)
    # model.layer3[-1].register_forward_hook(_forward_hook)
    model.avgpool.register_forward_hook(_forward_hook)
    model = model.to(device).eval()

    total_img_level_ROCAUC = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    center_nums = {'avgpool': args.k,}
    for class_name in MVTec_CLASS_NAMES:
        if class_name == "toothbrush":
            center_nums['avgpool'] = min(3, center_nums['avgpool'])

        trainset = MVTecDataset(root_path=args.data_root, is_train=True, class_name=class_name, resize=args.resize,
                                cropsize=args.crop_size)
        trainloader = DataLoader(trainset, batch_size=args.img_batch, shuffle=False, pin_memory=False)
        testset = MVTecDataset(root_path=args.data_root, is_train=False, class_name=class_name, resize=args.resize,
                               cropsize=args.crop_size)
        testloader = DataLoader(testset, batch_size=args.img_batch, shuffle=False, pin_memory=False)

        train_feas = OrderedDict([('avgpool', []), ])
        test_feas = OrderedDict([('avgpool', []), ])

        # 1. 提取训练集的特征
        train_feas_pkl = os.path.join(train_feas_path, f"train_{args.backbone}_{class_name}.pkl")
        # train_feas_pkl = os.path.join(temp_path, f"train_{args.backbone}_{class_name}.pkl")
        if not os.path.exists(train_feas_pkl):
            for x, y, mask in tqdm(trainloader, desc=f"[{class_name} train feature extract]"):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    model(x.to(device))
                for k, v in zip(train_feas, output_feas):
                    train_feas[k].append(v)
                output_feas = []
            for k, v in train_feas.items():
                train_feas[k] = torch.cat(v, 0)
            with open(train_feas_pkl, 'wb') as f:
                pickle.dump(train_feas, f)
        else:
            logger.info(f"load {train_feas_pkl}")
            with open(train_feas_pkl, 'rb') as f:
                train_feas = pickle.load(f)

        # testing
        test_criterion_pkl = os.path.join(temp_path, f"test_{args.backbone}_{class_name}_criterion.pkl")
        # if not os.path.exists(test_criterion_pkl):
        if True:
            torch.cuda.empty_cache()

            # Kmeans 聚类
            # center_nums = {'layer1': 3, 'layer2': 3, 'layer3': 3}
            kmeans_pkl = os.path.join(temp_path, f"test_{args.backbone}_{class_name}_gmm.pkl")
            if not os.path.exists(kmeans_pkl):
                kmeans_ = {}
                for k, v in train_feas.items():
                    logger.info(f"Model GMM {class_name} feature {k}...")
                    v = v.squeeze(-1).squeeze(-1)
                    # cpu
                    _means_, _vars_ = _kgaussians_fun_gpu(v, K=center_nums[k])

                    kmeans_[k] = {'mean': _means_,
                                  'var': _vars_}

                with open(kmeans_pkl, 'wb') as f:
                    pickle.dump(kmeans_, f)
            else:
                with open(kmeans_pkl, 'rb') as f:
                    kmeans_ = pickle.load(f)

            test_img_list = []
            test_y_list = []
            # 2. 提取测试集的特征
            for x, y, mask in tqdm(testloader, desc=f"[{class_name} test feature extract]"):
                torch.cuda.empty_cache()

                test_img_list.extend(x.detach().cpu().numpy())
                test_y_list.extend(y.detach().cpu().numpy())

                with torch.no_grad():
                    model(x.to(device))
                for k, v in zip(test_feas, output_feas):
                    test_feas[k].append(v)
                output_feas = []
            for k, v in test_feas.items():
                test_feas[k] = torch.cat(v, 0)

            # Image Level Anomaly Segmentation
            torch.cuda.empty_cache()
            test_feas_map = test_feas["avgpool"]
            test_feas_map = test_feas_map.squeeze(-1).squeeze(-1)

            _means_ = kmeans_["avgpool"]['mean']
            _vars_ = kmeans_["avgpool"]['var']
            _X_test = test_feas_map.to('cpu').numpy()

            dist_matrix_k = torch.zeros([test_feas_map.size(0), center_nums["avgpool"]])
            for k in range(center_nums["avgpool"]):
                _m = torch.from_numpy(_means_[k].reshape(-1).astype(np.float32)).to(device)
                _m = _m.unsqueeze(0)
                _inv = torch.from_numpy(np.linalg.inv(_vars_[k]).astype(np.float32)).to(device)
                # method 1
                delta = test_feas_map - _m
                temp = torch.mm(delta, _inv)
                dist_matrix_k[:, k] = torch.sqrt_(torch.sum(torch.mul(delta, temp), dim=1))
            # eeet = time.time()
            score_map_list = torch.min(dist_matrix_k, dim=1)[0]
            score_map_list = score_map_list.detach().cpu().numpy()

            torch.cuda.empty_cache()
            fpr, tpr, _ = roc_curve(test_y_list, score_map_list)
            img_level_ROCAUC = roc_auc_score(test_y_list, score_map_list)
            with open(test_criterion_pkl, 'wb') as f:
                pickle.dump([fpr, tpr, img_level_ROCAUC], f)
        else:
            with open(test_criterion_pkl, 'rb') as f:
                fpr, tpr, img_level_ROCAUC = pickle.load(f)
        logger.info('%s image ROCAUC: %.3f' % (class_name, img_level_ROCAUC))
        logger.info(f"\n")
        ax.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, img_level_ROCAUC))
        total_img_level_ROCAUC.append(img_level_ROCAUC)

    avg_img_level_ROCAUC = np.mean(np.array(total_img_level_ROCAUC))
    logger.info(f"Average image level ROCAUC: {avg_img_level_ROCAUC:.3f}")

    ax.title.set_text('Average image ROCAUC: %.3f' % np.mean(avg_img_level_ROCAUC))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, f'roc_curve_{args.backbone}.png'), dpi=100)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib as mpl

def euclidean_metric_np(X, centroids):
    n = X.shape[0]
    k = centroids.shape[0]
    X = np.expand_dims(X, 1)
    # X = X.unsqueeze(1).expand(n, k, -1)
    centroids = np.expand_dims(centroids, 0)
    # centroids = centroids.unsqueeze(0).expand(n, k, -1)
    dists = (X - centroids) ** 2
    dists = np.sum(dists, axis=2)
    return dists

def _kmeans_fun(X, K=10):
    _X = X.detach().cpu().numpy()
    D = _X.shape[1]
    _kmeans = KMeans(n_clusters=K, max_iter=1000, verbose=0, tol=1e-40)
    _kmeans.fit(_X)
    # logger.info(_kmeans.cluster_centers_)

    k_men = _kmeans.cluster_centers_
    k_var = np.zeros([K, D, D])

    _dist = euclidean_metric_np(_X, k_men)
    _idx_min = np.argmin(_dist, axis=1)
    for k in range(K):
        samples = _X[k == _idx_min]
        _m = np.mean(samples, axis=0)
        k_var[k] = LedoitWolf().fit(samples).covariance_
        # k_var[k] = np.cov(samples, rowvar=False)

    return k_men, k_var


def euclidean_metric(a, b):
    n = a.shape[0]
    k = b.shape[0]
    a = a.unsqueeze(1).expand(n, k, -1)
    b = b.unsqueeze(0).expand(n, k, -1)
    distance = ((a - b)**2).sum(dim=2)
    return distance

if __name__ == "__main__":
    main()
    logger.info(f"\n\n\n")
