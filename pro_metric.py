import numpy as np
from skimage import measure
from sklearn.metrics import auc
from loguru import logger


def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
    '''

    :param labeled_imgs: list, value 0/1
    :param score_imgs: list, float
    :param fpr_thresh: float
    :param max_steps: int
    :param class_name: str
    :return:
    '''
    n = len(labeled_imgs)
    labeled_imgs = np.array(labeled_imgs)
    # labeled_imgs[labeled_imgs<=0.45] = 0
    # labeled_imgs[labeled_imgs>0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.int)

    score_imgs = np.array(score_imgs)
    # print(labeled_imgs.shape)
    # print(score_imgs.shape)

    min_thresh = np.min(score_imgs)
    max_thresh = np.max(score_imgs)
    delta_tsh = (max_thresh - min_thresh) / max_steps

    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):

        thresh_step = min_thresh + delta_tsh * step
        threds.append(thresh_step)
        binary_score_maps[score_imgs <= thresh_step] = 0
        binary_score_maps[score_imgs > thresh_step] = 1

        pros_step = []
        for i in range(n):
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map, binary_score_maps[i])
            for prop in props:
                pros_step.append(prop.intensity_image.sum() / prop.area)
        # pro
        pros_mean.append(np.array(pros_step).mean())
        pros_std.append(np.array(pros_step).std())
        # fpr
        masks_neg = 1 - labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thresh_step)

    # default 30% fpr vs pro, pro_auc
    fprs = np.array(fprs)
    pros_mean = np.array(pros_mean)
    idx = fprs <= fpr_thresh  # # rescale fpr [0,0.3] -> [0, 1]

    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)

    pros_mean_selected = rescale(pros_mean[idx])  # need scale

    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    logger.info(f"{class_name} pro auc is: {pro_auc_score:.4f} at {fpr_thresh:.1f} FPR:")
    return pro_auc_score

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())