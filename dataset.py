"""
Dataset related functions
"""
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from utils import open_sequence

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)

		self.sequences = sequences

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index])

	def __len__(self):
		return len(self.sequences)


class ValDatasetDual(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder and in ground truth folder.
	"""
	def __init__(self, valsetdir=None, gt_dir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		gt_seqs_dirs = sorted(glob.glob(os.path.join(gt_dir, VALSEQPATT)))

		assert len(seqs_dirs) == len(gt_seqs_dirs), "Mismatch in number of validation input and ground truth sequences"

		# open individual sequences and append them to the sequence list
		sequences = []
		gt_sequences = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)

			gt_seq, _, _ = open_sequence(gt_seqs_dirs[seqs_dirs.index(seq_dir)], gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			gt_sequences.append(gt_seq)

			assert seq.shape == gt_seq.shape, "Mismatch in shape of input and ground truth sequences"
		
		self.sequences = sequences
		self.gt_sequences = gt_sequences

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), torch.from_numpy(self.gt_sequences[index])

	def __len__(self):
		return len(self.sequences)