import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from utils import batch_psnr, init_logger_test, rgb2y, close_logger, open_video
from statistics import mean
from vmaf_torch import VMAF
from pytorch_msssim import ssim, ms_ssim


def compute_metrics(denframes, seq_gt, vmaf, vmaf_neg, data_range=1.0, device="cpu"):
    psnr = batch_psnr(denframes, seq_gt, data_range=data_range)
    ssim_v = ssim(denframes, seq_gt, data_range=data_range, size_average=True)
    msssim_v = ms_ssim(denframes, seq_gt, data_range=data_range, size_average=True)
    
    seq_gt_y = rgb2y(seq_gt)
    denframes_y = rgb2y(denframes)
    vmaf_val = vmaf(seq_gt_y.to(device), denframes_y.to(device))
    vmaf_neg_val = vmaf_neg(seq_gt_y.to(device), denframes_y.to(device))

    return psnr, ssim_v, msssim_v, vmaf_val, vmaf_neg_val


def compute_metrics_on_dataset(input_dir, gt_dir, output_dir, device="cpu"):
    logger = init_logger_test(output_dir)
    device = device

    videos = [video for video in os.listdir(input_dir) if video.endswith('.mp4')]
    psnr_values = []
    ssim_values = []
    vmaf_values = []
    ms_ssim_values = []
    vmaf_neg_values = []
    vmaf = VMAF(temporal_pooling=True).to(device)
    vmaf_neg = VMAF(NEG=True, temporal_pooling=True).to(device)

    for video in videos:
        seq, _, _ = open_video(os.path.join(input_dir, video),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=20000000)
        seq_gt, _, _ = open_video(os.path.join(gt_dir, video),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=20000000)
            
        seq = torch.from_numpy(seq).to(device)
        seq_gt = torch.from_numpy(seq_gt).to(device)

        psnr_v, ssim_v, msssim_v, vmaf_v, vmaf_neg_v = compute_metrics(seq, seq_gt, vmaf, vmaf_neg, device=device)
        psnr_values.append(psnr_v.item())
        ssim_values.append(ssim_v.item())
        ms_ssim_values.append(msssim_v.item())
        vmaf_values.append(vmaf_v.item())
        vmaf_neg_values.append(vmaf_neg_v.item())

        print("PSNR: {}, SSIM: {}, MS-SSIM: {}, VMAF: {}, VMAF-NEG: {}".format(psnr_v, ssim_v, msssim_v, vmaf_v, vmaf_neg_v))

    logger.info(f"PSNR: {mean(psnr_values)}")
    logger.info(f"SSIM: {mean(ssim_values)}")
    logger.info(f"MS-SSIM: {mean(ms_ssim_values)}")
    logger.info(f"VMAF: {mean(vmaf_values)}")
    logger.info(f"VMAF-NEG: {mean(vmaf_neg_values)}")
    close_logger(logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for the denoiser")

    parser.add_argument("--input_dir", "-i", type=str, help="Image video path")
    parser.add_argument("--gt_dir", "-t", type=str, help="Ground truth video path")
    parser.add_argument("--output_dir", "-o", type=str, help="Output video path")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()

    compute_metrics_on_dataset(args.input_dir, args.gt_dir, args.output_dir, device=args.device)