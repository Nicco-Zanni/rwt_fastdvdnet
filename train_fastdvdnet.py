"""
Trains a FastDVDnet model.
"""
import wandb
import time
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from statistics import mean
from models import FastDVDnet
from dataset import ValDataset, ValDatasetDual
from dataloaders import train_dali_loader, train_dali_loader_dual
from utils import *
from train_common import resume_training, lr_scheduler, log_train_psnr, \
					validate_and_log, save_model_checkpoint
from noise_generator import smartphone_noise_generator, real_noise_generator, soft_noise_generator
from noise_generator.real_noise_config import train_real_noise_probabilities
from PIL import Image
from vmaf_torch import VMAF


def main(**args):
	r"""Performs the main training loop
	"""

	if args["wandb_log"]:
		wandb.init(
			project = "video-denoising",
			config = {
				"model": "fastdvdnet",
				"dataset": "DAVIS"
			}
		)

	# Load dataset
	print('> Loading datasets ...')
	
	if args["gt_dir"] is not None:
		# VAL 
		dataset_val = ValDatasetDual(valsetdir=args['valset_dir'], gt_dir=args["gt_val_dir"], gray_mode=False)
		loader_train = train_dali_loader_dual(batch_size=args['batch_size'],\
									input_root=args['trainset_dir'],\
									gt_root=args['gt_dir'],\
									sequence_length=args['temp_patch_size'],\
									crop_size=args['patch_size'],\
									epoch_size=args['max_number_patches'],\
									random_shuffle=True,\
									temp_stride=3)

	else:
		dataset_val = ValDataset(valsetdir=args['valset_dir'], gray_mode=False)
		for seq in dataset_val:
			print(seq.shape)
			break
		loader_train = train_dali_loader(batch_size=args['batch_size'],\
										file_root=args['trainset_dir'],\
										sequence_length=args['temp_patch_size'],\
										crop_size=args['patch_size'],\
										epoch_size=args['max_number_patches'],\
										random_shuffle=True,\
										temp_stride=3)

	num_minibatches = int(args['max_number_patches']//args['batch_size'])
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))

	# Init loggers
	writer, logger = init_logging(args)

	# Define GPU devices
	device_ids = [1]
	torch.backends.cudnn.benchmark = True # CUDNN optimization

	# Create model
	model = FastDVDnet(args["lightweight_model"], args["refine"], args["depthwise"])
	model = nn.DataParallel(model, device_ids=device_ids).cuda()

	# Define loss
	criterion = nn.MSELoss(reduction='sum')
	criterion.cuda()
	if args['vmaf_loss']:
		vmaf = VMAF(temporal_pooling=True).cuda()
	if args['vmaf_neg_loss']:
		vmaf_neg = VMAF(NEG=True, temporal_pooling=True).cuda()

	# Optimizer
	optimizer = optim.Adam(model.parameters(), lr=args['lr'])

	# Resume training or start anew
	start_epoch, training_params = resume_training(args, model, optimizer)

	# Training
	start_time = time.time()
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True

		# set learning rate in optimizer
		for param_group in optimizer.param_groups:
			param_group["lr"] = current_lr
		print('\nlearning rate %f' % current_lr)

		# train
		train_losses = []

		for i, data in enumerate(loader_train, 0):

			# Pre-training step
			model.train()

			# When optimizer = optim.Optimizer(net.parameters()) we only zero the optim's grads
			optimizer.zero_grad()

			img_train = data[0]['input'].to('cuda')  # [N, num_frames, C, H, W]

			if args["gt_dir"] is not None:
				gt_train = data[0]['ground_truth'].to('cuda')

				# Add Noise
				if args["noise_type"] == "smartphone":
					imgn_train = smartphone_noise_generator.generate_train_noisy_tensor(img_train, args["noise_gen_folder"], device=img_train.device)  # [N, F, C, H, W]
					img_train, imgn_train, gt_train = normalize_augment_gt(img_train, imgn_train, gt_train, ctrl_fr_idx)
				elif args["noise_type"] == "gaussian":
					raise ValueError("Gaussian noise not yet supported with ground truth")
				elif args["noise_type"] == "real":
					imgn_train, _ = real_noise_generator.apply_random_noise(img_train, train_real_noise_probabilities, batch=True, noise_gen_folder=args["noise_gen_folder"])
					img_train, imgn_train, gt_train = normalize_augment_gt(img_train, imgn_train, gt_train, ctrl_fr_idx)
				elif args["noise_type"] == "soft":
					raise ValueError("Soft noise not yet supported with ground truth")
				elif args["noise_type"] == "inherit":
					imgn_train = img_train.clone()
					img_train, imgn_train, gt_train = normalize_augment_gt(img_train, imgn_train, gt_train, ctrl_fr_idx)
				else:
					raise ValueError("Noise type not recognized")
			
			else:
				# Add Noise
				if args["noise_type"] == "smartphone":
					imgn_train = smartphone_noise_generator.generate_train_noisy_tensor(img_train, args["noise_gen_folder"], device=img_train.device)  # [N, F, C, H, W] [0, 255]
					img_train, imgn_train, gt_train = normalize_augment(img_train, imgn_train, ctrl_fr_idx) # [N, F*C, H, W] [0, 1]
				elif args["noise_type"] == "gaussian":
					img_train, gt_train = normalize_augment_clean(img_train, ctrl_fr_idx)
					noise = torch.zeros_like(img_train)
					noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
					imgn_train = img_train + noise
				elif args["noise_type"] == "real":
					imgn_train, _ = real_noise_generator.apply_random_noise(img_train, train_real_noise_probabilities, batch=True, noise_gen_folder=args["noise_gen_folder"])
					img_train, imgn_train, gt_train = normalize_augment(img_train, imgn_train, ctrl_fr_idx)
				elif args["noise_type"] == "soft":
					imgn_train = soft_noise_generator.add_soft_noise(img_train, sigma=random.randint(0, 5), gain=random.randint(2, 6), device=img_train.device)
					#imgn_train = video_compressor.compress_batch(imgn_train.contiguous().view(imgn_train.size()[0], args["temp_patch_size"], 3, imgn_train.size()[-2], imgn_train.size()[-1]))
					#imgn_train = imgn_train.contiguous().view(imgn_train.size()[0], -1, imgn_train.size()[-2], imgn_train.size()[-1])
					img_train, imgn_train, gt_train = normalize_augment(img_train, imgn_train, ctrl_fr_idx) # [N, F*C, H, W] [0, 1]
				elif args["noise_type"] == "inherit":
					raise ValueError("Inherit noise not supported without ground truth")
				else:
					raise ValueError("Noise type not recognized")

			# convert inp to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
			# extract ground truth (central frame)
			N, _, H, W = imgn_train.size()
			
			# Send tensors to GPU
			gt_train = gt_train.cuda(non_blocking=True)
			imgn_train = imgn_train.cuda(non_blocking=True)

			# Evaluate model and optimize it
			#out_train = model(imgn_train, noise_map)
			out_train = model(imgn_train)

			# Compute loss
			mse_loss = criterion(gt_train, out_train) / (N*2)
			vmaf_loss = 0
			vmaf_neg_loss = 0
			if args['vmaf_loss'] or args['vmaf_neg_loss']:
				gt_train_y = rgb2y(gt_train.cuda())
				out_train_y = rgb2y(out_train.cuda())

			if args['vmaf_loss']:
				vmaf_loss = 100 - vmaf(gt_train_y, out_train_y)
			if args['vmaf_neg_loss']:
				vmaf_neg_loss = 100 - vmaf_neg(gt_train_y, out_train_y)
			
			loss = args["mse_coef"] * mse_loss + args["vmaf_coef"] * vmaf_loss + args["vmaf_neg_coef"] * vmaf_neg_loss
			loss.backward()
			optimizer.step()
			train_losses.append(loss.item())

			# Results
			if training_params['step'] % args['save_every'] == 0:
				# Apply regularization by orthogonalizing filters
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)

				# Compute training PSNR
				log_train_psnr(out_train, \
								gt_train, \
								loss, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		# Call to model.eval() to correctly set the BN layers before inference
		model.eval()

		# Validation and log images
		psnr_val, ssim_val, ms_ssim_val, vmaf_val, vmaf_neg_val = validate_and_log(model_temp=model, \
																				dataset_val=dataset_val, \
																				valnoisestd=args['val_noiseL'], \
																				temp_psz=args['temp_patch_size'], \
																				writer=writer, \
																				epoch=epoch, \
																				lr=current_lr, \
																				logger=logger, \
																				trainimg=img_train, \
																				noise_gen_folder=args["noise_gen_folder"], \
																				noise_type=args["noise_type"], \
																				gt=True if args["gt_val_dir"] is not None else False)

		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer, training_params, epoch)

		if args["wandb_log"]:
			wandb.log({
				"Epoch": epoch,
				"LR": current_lr,
				"Train Loss (MSE)": mean(train_losses),
				"PSNR val": psnr_val,
				"SSIM val": ssim_val,
				"MS-SSIM val": ms_ssim_val,
				"VMAF val": vmaf_val,
				"VMAF NEG val": vmaf_neg_val
			})

	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

	if args["wandb_log"]:
		wandb.finish()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")

	#Training parameters
	parser.add_argument("--batch_size", type=int, default=64, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=60, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[40, 50], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr", type=float, default=1e-3, \
					 help="Initial learning rate")
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=10,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=5,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=96, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=64000, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default=None, \
					 help='path of trainset')
	parser.add_argument("--valset_dir", type=str, default=None, \
					 help='path of validation set')

	parser.add_argument("--noise_type", type=str, default='gaussian', choices=['gaussian', 'smartphone', 'real', 'soft', 'inherit'], help='type of noise')
	parser.add_argument("--noise_gen_folder", type=str, default="./noise_generator/", \
					 help='path of noise generator folder')

	# Light-weight Model
	parser.add_argument("--lightweight_model", action="store_true", help="Use a reduced model")

	# Refine Block and Depth-wise Convolutions
	parser.add_argument("--refine", action="store_true", help="Use a refine block at the end of the model")
	parser.add_argument("--depthwise", action="store_true", help="Use depthwise separable convolutions")

	# Ground truth
	parser.add_argument("--gt_dir", type=str, default=None, help="Path to ground truth images (default: input images)")
	parser.add_argument("--gt_val_dir", type=str, default=None, help="Path to ground truth images for validation (default: val images)")

	# VMAF Loss
	parser.add_argument("--vmaf_loss", action='store_true', help="Use VMAF loss")
	parser.add_argument("--vmaf_neg_loss", action='store_true', help="Use VMAF loss")
	parser.add_argument("--vmaf_coef", type=float, default='0.5', help="VMAF loss weight")
	parser.add_argument("--vmaf_neg_coef", type=float, default='0.5', help="VMAF NEG loss weight")
	parser.add_argument("--mse_coef", type=float, default='0.5', help="MSE loss weight")
	# WANDB
	parser.add_argument("--wandb_log", type=bool, default=False, help="Log in Weights & Biases")
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	if argspar.gt_dir is None and argspar.gt_val_dir is not None:
		raise ValueError("If gt_val_dir is provided, gt_dir must be provided as well")
	if argspar.gt_dir is not None and argspar.gt_val_dir is None:
		raise ValueError("If gt_dir is provided, gt_val_dir must be provided as well")

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
