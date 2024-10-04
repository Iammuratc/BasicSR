import argparse
import cv2
import numpy as np
import os
import torch


from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data.transforms import augment
from basicsr.utils.options import yaml_load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'  # noqa: E501
    )
    # parser.add_argument('--input', type=str, default='datasets/Set14/LRbicx4', help='input test image folder')
    # parser.add_argument('--gt', type=str, default='datasets/Set14/LRbicx4', help='ground truth test image folder')
    parser.add_argument('--output', type=str, default='results/ESRGAN', help='output folder')
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    # Read opt
    opt = yaml_load(args.opt)
    # Create the output folder
    os.makedirs(args.output, exist_ok=True)
    # Get file paths
    image_paths = sorted([os.path.join(opt['datasets']['val']['dataroot_lq'], file_path)
        for file_path in os.listdir(opt['datasets']['val']['dataroot_lq'])
        if os.path.isfile(os.path.join(opt['datasets']['val']['dataroot_lq'], file_path))])
    for path in image_paths:
        # imgname = os.path.splitext(os.path.basename(path))[0]
        imgname = os.path.basename(path)
        pred_folder = os.path.join(args.output,imgname)
        os.makedirs(pred_folder,exist_ok=True)
        print('Testing', imgname)
        # read input image
        # img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        # img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        # img = img.unsqueeze(0).to(device)
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(pred_folder, 'input_wo_noise.png'), img_lq)

        img_gt = cv2.imread(os.path.join(opt['datasets']['val']['dataroot_gt'],imgname))
        cv2.imwrite(os.path.join(pred_folder, 'gt_wo_noise.png'), img_gt)
        # Apply augmentations
        img_gt, img_lq = augment([img_gt, img_lq], opt=opt['datasets']['train'])
        cv2.imwrite(os.path.join(pred_folder, 'input.png'), img_lq)
        cv2.imwrite(os.path.join(pred_folder, 'gt.png'), img_gt)

        # Prepare the input image for the model
        img_lq = img_lq.astype(np.float32) / 255.
        img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_lq = img_lq.unsqueeze(0).to(device)

        # inference
        try:
            with torch.no_grad():
                output = model(img_lq)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(pred_folder, 'pred.png'), output)
        # break

if __name__ == '__main__':
    main()
