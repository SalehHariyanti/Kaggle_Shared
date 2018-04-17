'''
(Training examples, Test examples): ( 670 , 65 )
(256, 256, 3) : 334
(1024, 1024, 3) : 16
(520, 696, 3) : 92
(360, 360, 3) : 91
(512, 640, 3) : 13
(256, 320, 3) : 112
(1040, 1388, 3) : 1
(260, 347, 3) : 5
(603, 1272, 3) : 6
'''
import sys
sys.path.append('../')
import config
import numpy as np
np.random.seed(1234)
import os
import math
from settings import train_dir, test_dir, data_dir

from tensorflow.python.client import device_lib

def get_available_gpus():
    if os.name == 'nt':
        return ['gpu1']
    else:
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
   

class mask_rcnn_config(config.Config):

    def __init__(self, init_with = 'coco',
                 architecture = 'resnet50',
                 image_min_dim = 512, 
                 image_max_dim = 512, 
                 images_per_gpu = 2,
                 use_mini_mask = True,
                 max_gt_instances = 256,
                 mini_mask_shape = 56,
                 mask_shape = 28,
                 detection_nms_threshold = 0.3,
                 rpn_nms_threshold = 0.7,
                 steps = 600,
                 val_steps = 70,
                 train_data_root = train_dir,
                 val_data_root = train_dir,
                 test_data_root = test_dir,
                 model_dir = data_dir,
                 identifier = '',
                 augmentation_crop = 0.5,
                 augmentation_crop_min_size  = 256,
                 augmentation_crop_max_scale =   2,
                 augmentation_crop_min_scale = 0.8,
                 augmentation_dict = {'dim_ordering': 'tf', 'horizontal_flip': True, 'vertical_flip': True},
                 fn_load = 'load_image_gt_augment',
                 mask_size_dir = None):

        self.train_data_root = train_data_root
        self.val_data_root = val_data_root
        self.test_data_root = test_data_root
    
        self.MODEL_DIR = model_dir
        self.COCO_MODEL_PATH = os.path.join(base_dir, 'mask_rcnn_coco.h5')
        # imagenet, coco, or last
        self.init_with = init_with

        self.architecture = architecture
        
        self.identifier = identifier
    
        self.LEARNING_RATE = 1e-4
        self.LEARNING_MOMENTUM = 0.9
    
        # If enabled, resizes instance masks to a smaller size to reduce
        # memory load. Recommended when using high-resolution images.
        self.USE_MINI_MASK = use_mini_mask
        self.MINI_MASK_SHAPE = (mini_mask_shape, mini_mask_shape)  # (height, width) of the mini-mask
    
        self.GPU_COUNT = len(get_available_gpus())
        self.IMAGES_PER_GPU = images_per_gpu
         # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch.
        # typically be equal to the number of samples of your dataset divided by the batch size
        self.STEPS_PER_EPOCH = steps// self.BATCH_SIZE
        self.VALIDATION_STEPS = val_steps// self.BATCH_SIZE

        # Number of classes (including background)
        self.NUM_CLASSES = 1 + 1  # background + nucleis

        # Input image resing
        # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
        # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
        # be satisfied together the IMAGE_MAX_DIM is enforced.
        self.IMAGE_MIN_DIM = image_min_dim
        self.IMAGE_MAX_DIM = image_max_dim
        
        # Input image size
        self.IMAGE_SHAPE = np.array(
            [self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # If True, pad images with zeros such that they're (max_dim by max_dim)
        self.IMAGE_PADDING = True  # currently, the False option is not supported

        # Use smaller anchors because our image and objects are small
        self.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels, maybe add a 256?

        # The strides of each layer of the FPN Pyramid. These values
        # are based on a Resnet101 backbone.
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
                
        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride))]
             for stride in self.BACKBONE_STRIDES])

        # How many anchors per image to use for RPN training
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 320 #300
    
        # ROIs kept after non-maximum supression (training and inference)
        self.POST_NMS_ROIS_TRAINING = 2000
        self.POST_NMS_ROIS_INFERENCE = 2000

        # Pooled ROIs
        self.POOL_SIZE = mask_shape // 4
        self.MASK_POOL_SIZE = mask_shape // 2
        self.MASK_SHAPE = [mask_shape, mask_shape]

        # Number of ROIs per image to feed to classifier/mask heads
        # The Mask RCNN paper uses 512 but often the RPN doesn't generate
        # enough positive proposals to fill this and keep a positive:negative
        # ratio of 1:3. You can increase the number of proposals by adjusting
        # (increasing) the RPN NMS threshold.
        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        self.TRAIN_ROIS_PER_IMAGE = 512

        # Percent of positive ROIs used to train classifier/mask heads
        self.ROI_POSITIVE_RATIO = 0.33

        # Non-max suppression threshold to filter RPN proposals.
        # You can reduce(increase?) this during training to generate more propsals.
        self.RPN_NMS_THRESHOLD = rpn_nms_threshold

        # Maximum number of ground truth instances to use in one image
        self.MAX_GT_INSTANCES = max_gt_instances
       
        # Max number of final detections
        self.DETECTION_MAX_INSTANCES = 500 

        # Minimum probability value to accept a detected instance
        # ROIs below this threshold are skipped
        self.DETECTION_MIN_CONFIDENCE = 0.7 # may be smaller?
        # Non-maximum suppression threshold for detection
        self.DETECTION_NMS_THRESHOLD = detection_nms_threshold # 0.3    
    
        self.MEAN_PIXEL = np.array([0, 0, 0])
    
        # Weight decay regularization
        self.WEIGHT_DECAY = 0.0001

        # Ratios of anchors at each cell (width/height)
        # A value of 1 represents a square anchor, and 0.5 is a wide anchor
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

        # Anchor stride
        # If 1 then anchors are created for each cell in the backbone feature map.
        # If 2, then anchors are created for every other cell, and so on.
        self.RPN_ANCHOR_STRIDE = 1

        # Bounding box refinement standard deviation for RPN and final detections.
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        self.USE_RPN_ROIS = True

        self.augmentation_dict = augmentation_dict
        self.augmentation_crop = augmentation_crop
        self.augmentation_crop_min_size = augmentation_crop_min_size
        self.augmentation_crop_max_scale = augmentation_crop_max_scale
        self.augmentation_crop_min_scale = augmentation_crop_min_scale
        self.fn_load = fn_load

        self.NAME = self.get_name()

        # Set up mask size saving if requested
        if mask_size_dir is not None:
            self.mask_size_filename = os.path.join(mask_size_dir, ''.join((self.NAME, '.npy')))
            if not os.path.exists(self.mask_size_filename):
                mask_size = np.array([])
                np.save(self.mask_size_filename, mask_size)
        else:
            self.mask_size_filename = None


    def to_string(self, x):
        if isinstance(x, (list, dict)):
            _str = '-'.join([str(v) for v in x])
        else:
            _str = str(x)
        return _str


    def get_name(self):

        config_details = '_'.join(('DSB2018', 
                                   str(self.IMAGE_MIN_DIM), str(self.IMAGE_MAX_DIM), 
                                   str(self.USE_MINI_MASK), str(self.MINI_MASK_SHAPE[0]),
                                   str(self.MASK_SHAPE[0]),
                                   str(self.MAX_GT_INSTANCES), 
                                   str(self.DETECTION_NMS_THRESHOLD), 
                                   str(self.fn_load[-5:]),
                                   str(self.identifier)))

        if self.augmentation_dict is not None:
            aug_details = '-'.join(['-'.join((str(k[:5]), self.to_string(self.augmentation_dict[k])[:5])) for k in sorted(self.augmentation_dict.keys())])
            aug_details = '_'.join((aug_details, str(self.augmentation_crop)))
        else:
            aug_details = ''

        return '_'.join((config_details, aug_details))


def main():
    return

if __name__ == '__main__':
    main()
