# deep pretrained feature clustering for unsupervised anomaly detection

import os
import argparse
from loguru import logger
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision.models.resnet import wide_resnet50_2
# from dataset.mvtec import MVTecDataset, MVTec_CLASS_NAMES
from dataset.nanotwice import NanoDataset
from torch.utils.data import DataLoader

import time
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

def parse_args():
    parser = argparse.ArgumentParser('DPFC-GMM')

    parser.add_argument("--backbone", type=str, default='wide_resnet50_2')

    parser.add_argument("--img_batch", type=int, default=32)

    parser.add_argument("--time", type=int, default=3)
    parser.add_argument("--data_root", type=str, default="D:/Dataset/NanoTWICE/")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--save_path", type=str, default='result_nano')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train_feas_path = os.path.join(args.save_path, f'dpfc_kmeans_covariance-{args.crop_size}_{args.resize}', "temp")
    os.makedirs(train_feas_path, exist_ok=True)

    args.save_path = os.path.join(args.save_path, f'dpfc_kmeans_covariance-{args.crop_size}_{args.resize}', f"{args.backbone}", f"k_{args.k}")
    os.makedirs(args.save_path, exist_ok=True)
    temp_path = os.path.join(args.save_path, "temp")
    os.makedirs(temp_path, exist_ok=True)

    # logging
    args.logger = args.save_path + f'/logger-{args.time}.txt'
    logger.add(args.logger, rotation="200 MB", backtrace=True, diagnose=True)
    logger.info(str(args))

    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output_feas = []

    def _forward_hook(module, input, output):
        output_feas.append(output)

    # model = wide_resnet50_2(pretrained=True)
    model = eval(args.backbone)(pretrained=True)
    model.layer1[-1].register_forward_hook(_forward_hook)
    model.layer2[-1].register_forward_hook(_forward_hook)
    model.layer3[-1].register_forward_hook(_forward_hook)
    model = model.to(device).eval()

    total_pixel_level_ROCAUC = []
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    center_nums = {'layer1': args.k, 'layer2': args.k, 'layer3': args.k}

    trainset = NanoDataset(root_path=args.data_root, is_train=True, resize=args.resize, cropsize=args.crop_size)
    trainloader = DataLoader(trainset, batch_size=args.img_batch, shuffle=False, pin_memory=False)
    testset = NanoDataset(root_path=args.data_root, is_train=False, resize=args.resize, cropsize=args.crop_size)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=False)

    train_feas = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ])

    # 1. ????????????????????????
    train_feas_pkl = os.path.join(train_feas_path, f"train_{args.backbone}_nano.pkl")
    # train_feas_pkl = os.path.join(temp_path, f"train_{args.backbone}_{class_name}.pkl")
    if not os.path.exists(train_feas_pkl):
        for x, _ in tqdm(trainloader, desc=f"[Nano train feature extract]"):
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
    test_criterion_pkl = os.path.join(temp_path, f"test_{args.backbone}_Nano_criterion.pkl")
    # if not os.path.exists(test_criterion_pkl):
    if True:
        torch.cuda.empty_cache()

        # Kmeans ??????
        # center_nums = {'layer1': 3, 'layer2': 3, 'layer3': 3}
        kmeans_pkl = os.path.join(temp_path, f"test_{args.backbone}_Nano_gmm.pkl")
        # if not os.path.exists(kmeans_pkl):
        if True:
            kmeans_ = {}
            for k, v in train_feas.items():
                logger.info(f"Model GMM Nano feature {k}...")
                v = v.transpose(1, 3).flatten(0, 2)
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
        test_mask_list = []
        score_map_list = []
        # 2. ????????????????????????
        for x, mask in tqdm(testloader, desc=f"[Nano test feature extract]"):
            torch.cuda.empty_cache()
            test_feas = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ])

            test_img_list.extend(x.detach().cpu().numpy())
            test_mask_list.extend(mask.detach().cpu().numpy())

            with torch.no_grad():
                model(x.to(device))
            for k, v in zip(test_feas, output_feas):
                test_feas[k].append(v)
            output_feas = []

            for k, v in test_feas.items():
                test_feas[k] = torch.cat(v, 0)

            # for idx in tqdm(range(test_feas["layer1"].size(0)), desc=f"[{class_name} test pixel anomaly segment]"):
            idx = 0
            score_maps = []
            for _layer in test_feas.keys():
                torch.cuda.empty_cache()

                test_feas_map = test_feas[_layer][idx]
                _s = test_feas_map.size(1)
                test_feas_map = test_feas_map.transpose(0, 2).flatten(0, 1)
                _means_ = kmeans_[_layer]['mean']
                _vars_ = kmeans_[_layer]['var']
                ssst = time.time()
                dist_matrix_k = torch.zeros([test_feas_map.size(0), center_nums[_layer]])
                for k in range(center_nums[_layer]):
                    _m = torch.from_numpy(_means_[k].reshape(-1).astype(np.float32)).to(device)
                    _m = _m.unsqueeze(0)
                    _inv = torch.from_numpy(np.linalg.inv(_vars_[k]).astype(np.float32)).to(device)
                    # method 1
                    delta = test_feas_map - _m
                    temp = torch.mm(delta, _inv)
                    dist_matrix_k[:, k] = torch.sqrt_(torch.sum(torch.mul(delta, temp), dim=1))
                eeet = time.time()
                dist_matrix_k = torch.min(dist_matrix_k, dim=1)[0]
                score_map = torch.reshape(dist_matrix_k, (_s, _s)).transpose(0, 1)

                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=args.crop_size,
                                          mode='bilinear', align_corners=False)
                score_maps.append(score_map)
            # average distance between the features
            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
            # apply gaussian smoothing on the score map
            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            score_map_list.append(np.expand_dims(score_map, axis=0))

        flatten_mask_list = np.concatenate(test_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()
        torch.cuda.empty_cache()

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        # visualize(test_img_list, test_mask_list, score_map_list, threshold, class_name="Nano", save_path=args.save_path,
        #           num=1000)

        # calculate per-pixel level ROCAUC
        fpr, tpr, _ = roc_curve(flatten_mask_list, flatten_score_map_list)
        pixel_level_ROCAUC = roc_auc_score(flatten_mask_list, flatten_score_map_list)
        with open(test_criterion_pkl, 'wb') as f:
            pickle.dump([fpr, tpr, pixel_level_ROCAUC], f)
    else:
        with open(test_criterion_pkl, 'rb') as f:
            fpr, tpr, pixel_level_ROCAUC = pickle.load(f)

    logger.info(f"Average pixel level ROCAUC: {pixel_level_ROCAUC:.5f}")

    ax.plot(fpr, tpr, label='%s ROCAUC: %.5f' % ("Nano", pixel_level_ROCAUC))
    ax.title.set_text('Average pixel ROCAUC: %.5f' % pixel_level_ROCAUC)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, f'roc_curve_{args.backbone}.png'), dpi=100)




if __name__ == "__main__":
    main()
    logger.info(f"\n\n\n")
