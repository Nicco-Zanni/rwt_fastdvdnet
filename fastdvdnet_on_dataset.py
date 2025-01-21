#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.
"""
import os
import argparse
import time
import cv2
import torch
import torch.nn as nn
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet
from utils import batch_psnr, init_logger_test, rgb2y, \
                variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, open_video
from noise_generator import smartphone_noise_generator, real_noise_generator
from noise_generator.real_noise_config import test_real_noise_probabilities
from PIL import Image
import numpy as np
from statistics import mean
from vmaf_torch import VMAF
from pytorch_msssim import ssim, ms_ssim
from compute_metrics import compute_metrics


NUM_IN_FR_EXT = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqnoisy.size()[0]
    for idx in range(seq_len):
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir,\
                        ('n{}_{}').format(sigmaval, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir,\
                    ('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir,\
                    ('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)

def save_out_video(seqnoisy, seqclean, save_path, save_noisy, fps=30):
    """
    Saves the denoised and noisy sequences as a video under save_dir.

    Args:
        seqnoisy: torch.Tensor, sequence of noisy frames [num_frames, C, H, W] in [0, 1] range.
        seqclean: torch.Tensor, sequence of denoised frames [num_frames, C, H, W] in [0, 1] range.
        save_dir: string, directory to save the video.
        sigmaval: int/float, noise level indicator to include in video filename.
        suffix: string, additional label for the video filename.
        save_noisy: bool, if True, save a video with noisy frames alongside clean frames.
        fps: int, frames per second for the output video.
    """

    # Get video dimensions
    _, C, H, W = seqclean.size()
    assert C == 3, "Only 3-channel RGB videos are supported."

    # Define video writers
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Codec for MP4
    clean_writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    noisy_writer = None
    if save_noisy:
        save_noisy_path = save_path.replace(".mp4", "_noisy.mp4")
        noisy_writer = cv2.VideoWriter(save_noisy_path, fourcc, fps, (W, H))

    # Save frames to video
    seq_len = seqnoisy.size(0)
    for idx in range(seq_len):
        # Convert tensors to OpenCV-compatible images
        if save_noisy:
            noisy_frame = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            noisy_writer.write(noisy_frame)

        clean_frame = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        clean_writer.write(clean_frame)

    # Release video writers
    clean_writer.release()
    if noisy_writer:
        noisy_writer.release()

    print(f"Saved clean video: {save_path}")
    if save_noisy:
        print(f"Saved noisy video: {save_noisy_path}")


def denoise_dataset(**args):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

    Inputs:
        args (dict) fields:
            "model_file": path to model
            "test_path": path to sequence to denoise
            "suffix": suffix to add to output name
            "max_num_fr_per_seq": max number of frames to load per sequence
            "noise_sigma": noise level used on test set
            "dont_save_results: if True, don't save output images
            "no_gpu": if True, run model on CPU
            "save_path": where to save outputs as png
            "gray": if True, perform denoising of grayscale images instead of RGB
    """

    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create models
    print('Loading models ...')
    model_temp = FastDVDnet(args["lightweight_model"], refine=args["refine"], num_input_frames=NUM_IN_FR_EXT)

    # Load saved weights
    state_temp_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    runtimes = []
    psnrs = []
    ssims = []
    mssims = []
    vmafs = []
    vmaf_negs = []
    vmaf = VMAF(temporal_pooling=True)
    vmaf_neg = VMAF(NEG=True, temporal_pooling=True)
    
    if args['data_type'] == 'video':
        videos = [v for v in os.listdir(args["test_dir"]) if v.endswith(('.mp4', '.avi', '.mov'))]
        for video in videos:
            # process data
            open_seq_time = time.time()
            seq, _, _ = open_video(os.path.join(args['test_dir'], video),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=args['max_num_fr_per_seq'])
            seq_gt, _, _ = open_video(os.path.join(args['gt_dir'], video),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=args['max_num_fr_per_seq'])
            
            seq = torch.from_numpy(seq).to(device)
            seq_gt = torch.from_numpy(seq_gt).to(device)

            open_seq_time = time.time() - open_seq_time

            print("Denoising...")
            denoise_time = time.time()
            with torch.no_grad():
                denframes = denoise_seq_fastdvdnet(seq=seq,\
                                                temp_psz=NUM_IN_FR_EXT,\
                                                model_temporal=model_temp)
            denoise_time = time.time() - denoise_time

            print("Computing metrics...")
            # Compute Metrics
            psnr_v, ssim_v, msssim_v, vmaf_v, vmaf_neg_v = compute_metrics(denframes, seq_gt, vmaf, vmaf_neg, data_range=1.0)
            psnrs.append(psnr_v.item())
            ssims.append(ssim_v.item())
            mssims.append(msssim_v.item())
            vmafs.append(vmaf_v.item())
            vmaf_negs.append(vmaf_neg_v.item())
            runtimes.append((open_seq_time, denoise_time))

            print("Saving outputs...")
            # Save outputs
            save_out_video(seq, denframes, os.path.join(args['save_path'], video), args['save_noisy'])

    else:
        sequences = [seq for seq in os.listdir(args['test_dir']) if os.path.isdir(os.path.join(args['test_dir'], seq))]
        for sequence in sequences:
            # process data
            open_seq_time = time.time()
            seq, _, _ = open_sequence(os.path.join(args['test_dir'], sequence),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=args['max_num_fr_per_seq'])
            seq_gt, _, _ = open_sequence(os.path.join(args['gt_dir'], sequence),\
                                    gray_mode=False,\
                                    expand_if_needed=False,\
                                    max_num_fr=args['max_num_fr_per_seq'])
            
            seq = torch.from_numpy(seq).to(device)
            seq_gt = torch.from_numpy(seq_gt).to(device)

            open_seq_time = time.time() - open_seq_time

            print("Denoising...")
            denoise_time = time.time()
            with torch.no_grad():
                denframes = denoise_seq_fastdvdnet(seq=seq,\
                                                temp_psz=NUM_IN_FR_EXT,\
                                                model_temporal=model_temp)
            denoise_time = time.time() - denoise_time

            psnr, ssim, msssim, vmaf, vmaf_neg = compute_metrics(denframes, seq_gt, data_range=1.0)
            psnrs.append(psnr)
            ssims.append(ssim)
            mssims.append(msssim)
            vmafs.append(vmaf)
            vmaf_negs.append(vmaf_neg)
            runtimes.append((open_seq_time, denoise_time))

            save_out_video(seq, denframes, os.path.join(args['save_path'], sequence + ".mp4"), args['save_noisy'])

    # Compute Average Metrics
    avg_psnr = mean(psnrs)
    avg_ssim = mean(ssims)
    avg_msssim = mean(mssims)
    avg_vmaf = mean(vmafs)
    avg_vmaf_neg = mean(vmaf_negs)
    avg_seq_time = mean(runtimes[:][0])
    avg_denoise_time = mean(runtimes[:][1])

    logger.info(f"Average PSNR: {avg_psnr:.2f}")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Average MS-SSIM: {avg_msssim:.4f}")
    logger.info(f"Average VMAF: {avg_vmaf:.2f}")
    logger.info(f"Average VMAF_NEG: {avg_vmaf_neg:.2f}")
    logger.info(f"Average Sequence Open Time: {avg_seq_time:.2f}")
    logger.info(f"Average Denoise Time: {avg_denoise_time:.2f}")

    close_logger(logger)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,default="./model.pth", help='path to model of the pretrained denoiser')

    parser.add_argument("--lightweight_model", action='store_true', help='use lightweight FastDVDnet model')
    parser.add_argument("--refine", action="store_true", help="Use a refine block at the end of the model")

    parser.add_argument("--test_dir", type=str, default="./data/rgb/Kodak24", help='path to sequence to denoise')

    parser.add_argument("--gt_dir", type=str, default=None, help='path to GT sequence to compute PSNR (default: clean input)')

    parser.add_argument("--data_type", type=str, default="video", choices=["video", "images"], help='denoise videos or image sequences')

    parser.add_argument("--max_num_fr_per_seq", type=int, default=200, help='max number of frames to load per sequence')

    parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='./results', help='where to save outputs')


    argspar = parser.parse_args()
    # Normalize noises ot [0, 1]

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    denoise_dataset(**vars(argspar))
