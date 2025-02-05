'''Implements a sequence dataloader using NVIDIA's DALI library.

The dataloader is based on the VideoReader DALI's module, which is a 'GPU' operator that loads
and decodes H264 video codec with FFmpeg.

Based on
https://github.com/NVIDIA/DALI/blob/master/docs/examples/video/superres_pytorch/dataloading/dataloaders.py
'''
import os
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class VideoReaderPipeline(Pipeline):
	''' Pipeline for reading H264 videos based on NVIDIA DALI.
	Returns a batch of sequences of `sequence_length` frames of shape [N, F, C, H, W]
	(N being the batch size and F the number of frames). Frames are RGB uint8.
	Args:
		batch_size: (int)
				Size of the batches
		sequence_length: (int)
				Frames to load per sequence.
		num_threads: (int)
				Number of threads.
		device_id: (int)
				GPU device ID where to load the sequences.
		files: (str or list of str)
				File names of the video files to load.
		crop_size: (int)
				Size of the crops. The crops are in the same location in all frames in the sequence
		random_shuffle: (bool, optional, default=True)
				Whether to randomly shuffle data.
		step: (int, optional, default=-1)
				Frame interval between each sequence (if `step` < 0, `step` is set to `sequence_length`).
	'''
	def __init__(self, batch_size, sequence_length, num_threads, device_id, files,
				 crop_size, random_shuffle=True, step=-1):
		super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
		# Define VideoReader
		self.reader = ops.VideoReader(device="gpu",
										filenames=files,
										sequence_length=sequence_length,
										normalized=False,
										random_shuffle=random_shuffle,
										image_type=types.DALIImageType.RGB,
										dtype=types.DALIDataType.UINT8,
										step=step,
										initial_fill=16)

		# Define crop and permute operations to apply to every sequence
		self.crop = ops.CropMirrorNormalize(device="gpu",
										crop_w=crop_size,
										crop_h=crop_size,
										output_layout='FCHW',
										dtype=types.DALIDataType.FLOAT)
		self.uniform = ops.Uniform(range=(0.0, 1.0))  # used for random crop

	def define_graph(self):
		'''Definition of the graph--events that will take place at every sampling of the dataloader.
		The random crop and permute operations will be applied to the sampled sequence.
		'''
		input = self.reader(name="Reader")
		cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
		return cropped


class train_dali_loader():
	'''Sequence dataloader.
	Args:
		batch_size: (int)
			Size of the batches
		file_root: (str)
			Path to directory with video sequences
		sequence_length: (int)
			Frames to load per sequence
		crop_size: (int)
			Size of the crops. The crops are in the same location in all frames in the sequence
		epoch_size: (int, optional, default=-1)
			Size of the epoch. If epoch_size <= 0, epoch_size will default to the size of VideoReaderPipeline
		random_shuffle (bool, optional, default=True)
			Whether to randomly shuffle data.
		temp_stride: (int, optional, default=-1)
			Frame interval between each sequence
			(if `temp_stride` < 0, `temp_stride` is set to `sequence_length`).
	'''
	def __init__(self, batch_size, file_root, sequence_length,
				 crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
		# Builds list of sequence filenames
		container_files = os.listdir(file_root)
		container_files = [file_root + '/' + f for f in container_files]
		# Define and build pipeline
		self.pipeline = VideoReaderPipeline(batch_size=batch_size,
											sequence_length=sequence_length,
											num_threads=2,
											device_id=1,
											files=container_files,
											crop_size=crop_size,
											random_shuffle=random_shuffle,
											step=temp_stride)
		self.pipeline.build()

		# Define size of epoch
		if epoch_size <= 0:
			self.epoch_size = self.pipeline.epoch_size("Reader")
		else:
			self.epoch_size = epoch_size
		self.dali_iterator = pytorch.DALIGenericIterator(pipelines=self.pipeline,
														output_map=["input"],
														size=self.epoch_size,
														auto_reset=True)

	def __len__(self):
		return self.epoch_size

	def __iter__(self):
		return self.dali_iterator.__iter__()


class VideoReaderPipelineDual(Pipeline):
    '''Pipeline for reading H264 videos from two directories and creating paired patches.
    Args:
        batch_size: (int)
            Size of the batches
        sequence_length: (int)
            Frames to load per sequence
        num_threads: (int)
            Number of threads
        device_id: (int)
            GPU device ID
        input_files: (list of str)
            File names of input video files
        gt_files: (list of str)
            File names of ground truth video files
        crop_size: (int)
            Size of the crops
        random_shuffle: (bool, optional, default=True)
            Whether to randomly shuffle data
        step: (int, optional, default=-1)
            Frame interval between each sequence
    '''
    def __init__(self, batch_size, sequence_length, num_threads, device_id, input_files, gt_files,
                 crop_size, random_shuffle=True, seed=12, step=-1):
        super(VideoReaderPipelineDual, self).__init__(batch_size, num_threads, device_id, seed=seed, prefetch_queue_depth=2)
        
        # VideoReader for input videos
        self.input_reader = ops.VideoReader(
            device="gpu",
            filenames=input_files,
            sequence_length=sequence_length,
            normalized=False,
            random_shuffle=random_shuffle,
            image_type=types.DALIImageType.RGB,
            dtype=types.DALIDataType.UINT8,
            step=step,
            initial_fill=32,
			seed=seed
        )
        
        # VideoReader for ground truth videos
        self.gt_reader = ops.VideoReader(
            device="gpu",
            filenames=gt_files,
            sequence_length=sequence_length,
            normalized=False,
            random_shuffle=random_shuffle,
            image_type=types.DALIImageType.RGB,
            dtype=types.DALIDataType.UINT8,
            step=step,
            initial_fill=32,
			seed=seed
        )
        
        # Crop and normalize operations
        self.crop = ops.CropMirrorNormalize(
            device="gpu",
            crop_w=crop_size,
            crop_h=crop_size,
            output_layout='FCHW',
            dtype=types.DALIDataType.FLOAT
        )
        self.uniform = ops.Uniform(range=(0.0, 1.0))  # used for random crop

    def define_graph(self):
        '''Define the data processing graph.'''
        input_sequence = self.input_reader(name="InputReader")
        gt_sequence = self.gt_reader(name="GTReader")

        crop_pos_x = self.uniform()
        crop_pos_y = self.uniform()
        # Apply random crop to both input and ground truth sequences
        input_cropped = self.crop(input_sequence, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        gt_cropped = self.crop(gt_sequence, crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
        
        return input_cropped, gt_cropped


class train_dali_loader_dual():
    '''Dataloader for paired sequences from two video directories.
    Args:
        batch_size: (int)
            Size of the batches
        input_root: (str)
            Path to directory with input video sequences
        gt_root: (str)
            Path to directory with ground truth video sequences
        sequence_length: (int)
            Frames to load per sequence
        crop_size: (int)
            Size of the crops
        epoch_size: (int, optional, default=-1)
            Size of the epoch
        random_shuffle: (bool, optional, default=True)
            Whether to randomly shuffle data
        temp_stride: (int, optional, default=-1)
            Frame interval between each sequence
    '''
    def __init__(self, batch_size, input_root, gt_root, sequence_length,
                 crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1):
        # Get list of video filenames for input and ground truth
        input_files = sorted([os.path.join(input_root, f) for f in os.listdir(input_root)])
        gt_files = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)])
        
        # Ensure input and ground truth files are paired
        assert len(input_files) == len(gt_files), "Mismatch in number of input and ground truth videos"
        for inp, gt in zip(input_files, gt_files):
            assert os.path.basename(inp) == os.path.basename(gt), \
                f"Unmatched files: {inp} and {gt}"
        
        # Define and build the dual pipeline
        self.pipeline = VideoReaderPipelineDual(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=min(2, os.cpu_count()),
            device_id=1,
            input_files=input_files,
            gt_files=gt_files,
            crop_size=crop_size,
            random_shuffle=random_shuffle,
            step=temp_stride
        )
        self.pipeline.build()

        # Define size of epoch
        if epoch_size <= 0:
            self.epoch_size = self.pipeline.epoch_size("InputReader")
        else:
            self.epoch_size = epoch_size
        
        # DALI iterator for paired sequences
        self.dali_iterator = pytorch.DALIGenericIterator(
            pipelines=self.pipeline,
            output_map=["input", "ground_truth"],
            size=self.epoch_size,
            auto_reset=True
        )

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        return iter(self.dali_iterator)
