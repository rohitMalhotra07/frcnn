from keras import backend as K
import math

class Config:

	def __init__(self):
		self.verbose = True

		self.network = 'resnet50'

		# setting for data augmentation
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False

		# anchor box scales
		self.anchor_box_scales = [128, 256, 512]

		# anchor box ratios
		self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

		# size to resize the smallest side of the image
		self.im_size = 600

		# image channel-wise mean to subtract
		self.img_channel_mean = [103.939, 116.779, 123.68]
		self.img_scaling_factor = 1.0

		# number of ROIs at once
		self.num_rois = 4

		# stride at the RPN (this depends on the network configuration)
		self.rpn_stride = 16

		self.balanced_classes = False

		# scaling the stdev
		self.std_scaling = 4.0
		self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

		# overlaps for RPN
		self.rpn_min_overlap = 0.3
		self.rpn_max_overlap = 0.7

		# overlaps for classifier ROIs
		self.classifier_min_overlap = 0.1
		self.classifier_max_overlap = 0.5

		# placeholder for the class mapping, automatically generated by the parser
		self.class_mapping = {'bg': 0,
 'person': 1,
 'bicycle': 2,
 'car': 3,
 'motorcycle': 4,
 'airplane': 5,
 'bus': 6,
 'train': 7,
 'truck': 8,
 'boat': 9,
 'traffic light': 10,
 'fire hydrant': 11,
 'stop sign': 12,
 'parking meter': 13,
 'bench': 14,
 'bird': 15,
 'cat': 16,
 'dog': 17,
 'horse': 18,
 'sheep': 19,
 'cow': 20,
 'elephant': 21,
 'bear': 22,
 'zebra': 23,
 'giraffe': 24,
 'backpack': 25,
 'umbrella': 26,
 'handbag': 27,
 'tie': 28,
 'suitcase': 29,
 'frisbee': 30,
 'skis': 31,
 'snowboard': 32,
 'sports ball': 33,
 'kite': 34,
 'baseball bat': 35,
 'baseball glove': 36,
 'skateboard': 37,
 'surfboard': 38,
 'tennis racket': 39,
 'bottle': 40,
 'wine glass': 41,
 'cup': 42,
 'fork': 43,
 'knife': 44,
 'spoon': 45,
 'bowl': 46,
 'banana': 47,
 'apple': 48,
 'sandwich': 49,
 'orange': 50,
 'broccoli': 51,
 'carrot': 52,
 'hot dog': 53,
 'pizza': 54,
 'donut': 55,
 'cake': 56,
 'chair': 57,
 'couch': 58,
 'potted plant': 59,
 'bed': 60,
 'dining table': 61,
 'toilet': 62,
 'tv': 63,
 'laptop': 64,
 'mouse': 65,
 'remote': 66,
 'keyboard': 67,
 'cell phone': 68,
 'microwave': 69,
 'oven': 70,
 'toaster': 71,
 'sink': 72,
 'refrigerator': 73,
 'book': 74,
 'clock': 75,
 'vase': 76,
 'scissors': 77,
 'teddy bear': 78,
 'hair drier': 79,
 'toothbrush': 80}


		#location of pretrained weights for the base network 
		# weight files can be found at:
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
		# https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

		self.model_path = 'model_frcnn.vgg.hdf5'
