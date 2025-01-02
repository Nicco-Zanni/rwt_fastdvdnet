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
from utils import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, open_video
from noise_generator import smartphone_noise_generator, real_noise_generator
from noise_generator.real_noise_config import test_real_noise_probabilities
from PIL import Image
import numpy as np


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

def save_out_video(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy, fps=30):
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
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get video dimensions
    _, C, H, W = seqclean.size()
    assert C == 3, "Only 3-channel RGB videos are supported."

    # Define file names
    if len(suffix) == 0:
        out_video_name = os.path.join(save_dir, f'n{sigmaval}_FastDVDnet.mp4')
    else:
        out_video_name = os.path.join(save_dir, f'n{sigmaval}_FastDVDnet_{suffix}.mp4')

    if save_noisy:
        noisy_video_name = os.path.join(save_dir, f'n{sigmaval}_noisy.mp4')

    # Define video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    clean_writer = cv2.VideoWriter(out_video_name, fourcc, fps, (W, H))
    noisy_writer = None
    if save_noisy:
        noisy_writer = cv2.VideoWriter(noisy_video_name, fourcc, fps, (W, H))

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

    print(f"Saved clean video: {out_video_name}")
    if save_noisy:
        print(f"Saved noisy video: {noisy_video_name}")

def test_fastdvdnet(**args):
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
	model_temp = FastDVDnet(args["lightweight_model"], num_input_frames=NUM_IN_FR_EXT)

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

	with torch.no_grad():
		# process data
		open_seq_time = time.time()
		if args['video']:
			seq, _, _ = open_video(args['test_path'],\
									args['gray'],\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
			if args['gt_path'] is not None:
				seq_gt, _, _ = open_video(args['gt_path'],\
										args['gray'],\
										expand_if_needed=False,\
										max_num_fr=args['max_num_fr_per_seq'])
		else:
			seq, _, _ = open_sequence(args['test_path'],\
										args['gray'],\
										expand_if_needed=False,\
										max_num_fr=args['max_num_fr_per_seq'])
			if args['gt_path'] is not None:
				seq_gt, _, _ = open_sequence(args['gt_path'],\
										args['gray'],\
										expand_if_needed=False,\
										max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq).to(device)

		open_seq_time = time.time() - open_seq_time

		# Add noise
		if args['noise_type'] == 'smartphone':
			noise_type = 'smartphone'
			seqn = smartphone_noise_generator.generate_val_noisy_tensor(seq, args['noise_gen_folder'], device=seq.device).squeeze(0)
		
		elif args['noise_type'] == 'gaussian':
			noise_type = 'gaussian'
			noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device)
			seqn = seq + noise
		
		elif args['noise_type'] == 'real':
			seqn, noise_type = real_noise_generator.apply_random_noise(seq, test_real_noise_probabilities, batch=False, noise_gen_folder=args['noise_gen_folder'])
		else:
			raise ValueError("Noise type not recognized")

		'''
		seq_img = (seqn[0]).to('cpu').numpy().astype(np.uint8)
		seq_img = seq_img.transpose(1, 2, 0)
		seq_img = Image.fromarray(seq_img)
		seq_img.save("results/after_noise.png")
		'''
		print("Denoising...")
		denoise_time = time.time()
		denframes = denoise_seq_fastdvdnet(seq=seqn,\
										temp_psz=NUM_IN_FR_EXT,\
										model_temporal=model_temp)
		denoise_time = time.time() - denoise_time

	if args['gt_path'] is None:
		seq_gt = seq
	seq_gt.to(seq.device)
	# Compute PSNR and log it
	runtime = open_seq_time + denoise_time
	psnr = batch_psnr(denframes, seq_gt, 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq_gt, 1.)
	seq_length = seq.size()[0]
	logger.info("Finished denoising {}".format(args['test_path']))
	logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s, Total running time: {:.3f}".\
				 format(seq_length, denoise_time, open_seq_time, runtime))
	logger.info("\tNoise type: {}: {}".format(args['noise_type'], noise_type))
	logger.info("\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, psnr))

	# Save outputs
	if not args['dont_save_results']:
		if args["video"]:
			save_out_video(seqn, denframes, args['save_path'], int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])
		else:
			save_out_seq(seqn, denframes, args['save_path'], int(args['noise_sigma']*255), args['suffix'], args['save_noisy'])

	# close logger
	close_logger(logger)

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,\
						default="./model.pth", \
						help='path to model of the pretrained denoiser')

	parser.add_argument("--lightweight_model", action='store_true', help='use lightweight FastDVDnet model')

	parser.add_argument("--test_path", type=str, default="./data/rgb/Kodak24", \
						help='path to sequence to denoise')

	parser.add_argument("--gt_path", type=str, default=None, help='path to GT sequence to compute PSNR (default: clean input)')

	parser.add_argument("--video", action='store_true', help='denoise video files instead of sequences')
	parser.add_argument("--multiple", action='store_true', help='denoise multiple sequences or videos')

	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=100000000, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')
	parser.add_argument("--gray", action='store_true',\
						help='perform denoising of grayscale images instead of RGB')

	# Custom noise
	parser.add_argument("--noise_type", type=str, default='gaussian', choices=['gaussian', 'smartphone', 'real'], help='type of noise')
	parser.add_argument("--noise_gen_folder", type=str, default="./noise_generator/", \
					 help='path of noise generator folder')

	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_fastdvdnet(**vars(argspar))
