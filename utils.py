import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance_matrix(x, y):
    m = x.size(0)
    n = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(m, n, d)
    y = y.unsqueeze(0).expand(m, n, d)

    dist_matrix = torch.sqrt(torch.pow(x - y, 2)).sum(dim=2)
    return dist_matrix


def visualize(test_imgs, test_masks, predict_masks, threshold, class_name, save_path, num=10):
    num = min(num, len(test_imgs))
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    for _idx in range(num):
        test_img = test_imgs[_idx]
        test_img = denormalize(test_img)
        test_mask = test_masks[_idx].transpose(1, 2, 0).squeeze()
        pred_mask = predict_masks[_idx]
        if len(pred_mask.shape) == 3:
            pred_mask = np.reshape(pred_mask, (pred_mask.shape[1], pred_mask.shape[2]))
        pred_mask[pred_mask <= threshold] = 0
        pred_mask[pred_mask > threshold] = 1
        # test_pred_img = test_img.copy()
        # test_pred_img[pred_mask == 0] = 0

        paints = draw_detect(test_img, test_mask, pred_mask)
        _path1 = os.path.join(save_path, 'images', '%s_%03d_res.png' % (class_name, _idx))
        cv2.imwrite(_path1, paints)
        _path0 = os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, _idx))

        _tt = np.uint8(test_img)
        _tt = cv2.cvtColor(_tt, cv2.COLOR_RGB2BGR)
        cv2.imwrite(_path0, _tt)

        # fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        # fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #
        # for ax_i in ax_img:
        #     ax_i.axes.xaxis.set_visible(False)
        #     ax_i.axes.yaxis.set_visible(False)
        #
        # ax_img[0].imshow(test_img)
        # ax_img[0].title.set_text('Image')
        # ax_img[1].imshow(test_mask, cmap='gray')
        # ax_img[1].title.set_text('GroundTruth')
        # ax_img[2].imshow(pred_mask, cmap='gray')
        # ax_img[2].title.set_text('Predict mask')
        # ax_img[3].imshow(test_pred_img)
        # ax_img[3].title.set_text('Predict anomalous image')
        #
        # os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        # fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, _idx)), dpi=100)
        # fig_img.clf()
        # plt.close(fig_img)

def draw_detect(img, label, binary):
    assert len(label.shape) == 2
    assert len(binary.shape) == 2
    label = np.uint8(label*255)
    binary = np.uint8(binary*255)
    # img = np.transpose(img, (2, 1, 0))
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, label = cv2.threshold(label, 5, 255, 0)
    _, label_cnts, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    _, binary = cv2.threshold(binary, 5, 255, 0)
    _, binary_cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape)
    mask = cv2.fillPoly(mask, binary_cnts, color=(0, 255, 0))
    img = img + 0.4 * mask
    img = np.clip(img, 0, 255)
    img = np.uint8(img)

    img = cv2.drawContours(img, label_cnts, -1, (0, 0, 255), 2)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

def denormalize(img):
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    x = (((img.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

import os
from PIL import Image

import torch
from torchvision import transforms as T
def load_image(path):
    transform_x = T.Compose([T.Resize(256, Image.ANTIALIAS),
                             T.CenterCrop(224),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])])
    x = Image.open(path).convert('RGB')
    x = transform_x(x)
    x = x.unsqueeze(0)
    return x