import os
import torch
import argparse
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, resnet50, wide_resnet50_2, wide_resnet101_2, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
import torchsummary
from thop import profile
from thop import clever_format

def parse_args():
    parser = argparse.ArgumentParser('DPFC-GMM')

    parser.add_argument("--backbone", type=str, default='resnet50')

    parser.add_argument("--img_batch", type=int, default=32)
    parser.add_argument("--fea_batch", type=int, default=128)

    parser.add_argument("--data_root", type=str, default="D:/Dataset/mvtec_anomaly_detection/")
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)

    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--save_path", type=str, default='result')
    # parser.add_argument("--save_path", type=str, default='result')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    output_feas = []

    def _forward_hook(module, input, output):
        output_feas.append(output)

    model = eval(args.backbone)(pretrained=False)
    model.layer1[-1].register_forward_hook(_forward_hook)
    model.layer2[-1].register_forward_hook(_forward_hook)
    model.layer3[-1].register_forward_hook(_forward_hook)
    model = model.to(device).eval()

    x = torch.rand([1, 3, 256, 256]).to(device)
    # y = model(x)
    # torchsummary.summary(model, (3, 224, 224))

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"[INFO] {args.backbone}-flops: {flops}")
    print(f"[INFO] {args.backbone}-params: {params}")
    for i, feas in enumerate(output_feas):
        print(f"Layer {i+1} size is: {feas.shape}")


if __name__ == "__main__":
    main()