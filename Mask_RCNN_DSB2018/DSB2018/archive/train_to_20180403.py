import os
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
np.random.seed(1234)
import model as modellib
from model import log
import utils
import random
from settings import train_dir, supplementary_dir


def load_weights(model, _config, init_with_override = None):

    init_with = _config.init_with if init_with_override is None else init_with_override  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        if not os.path.exists(_config.COCO_MODEL_PATH):
            utils.download_trained_weights(_config.COCO_MODEL_PATH)
    
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(_config.COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    return model


def train_resnet101_flips_by_colour_inc_suppdata():

    ################
    # Greys first 
    ################

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_1')

    # Training dataset
    dataset_train = DSB2018_Dataset(_config.train_data_root)
    dataset_train.add_nuclei(_config.train_data_root, 'train', target_colour_id = np.array([1]))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset(_config.train_data_root)
    dataset_val.add_nuclei(_config.val_data_root, 'val', target_colour_id = np.array([1]))
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


    ################
    # Colour second
    ################

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_2')

    # Training dataset
    dataset_train = DSB2018_Dataset(_config.train_data_root)
    dataset_train.add_nuclei(_config.train_data_root, 'train', target_colour_id = np.array([2]))
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset(_config.train_data_root)
    dataset_val.add_nuclei(_config.val_data_root, 'val', target_colour_id = np.array([2]))
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val')
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_yblur():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'y_gaussian_blur': [-0.25, 0.25]})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train')
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val')
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_augcrop0p8():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               augmentation_crop = 0.8)

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_maxgt400_minimaskFalse():

    _config = mask_rcnn_config(init_with = 'coco',
                               max_gt_instances = 400,
                               use_mini_mask = False,
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_no_aug():

    _config = mask_rcnn_config(augmentation_dict = {'dim_ordering': 'tf'})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.9)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.9)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE / 10,
                epochs=50,
                layers='all')


def train_resnet101_flips_alldata_minimask28():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 28,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_maxgt196():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               max_gt_instances = 196,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask18():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 18,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_maxgt128():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               max_gt_instances = 128,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask12():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_minimask12_maxgt196():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               max_gt_instances = 196,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask8():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 8,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask12_mask16():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               mask_shape = 16,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_minimask12_augcontrast():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'contrast_stretching': 10})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_minimask12_ep50():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=50,
                layers='all')

def train_resnet101_flips_alldata_minimask12_size768():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               image_max_dim = 768,
                               image_min_dim = 768, 
                               images_per_gpu = 1,                              
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')

def train_resnet101_flips_alldata_minimask12_detectionnms0_5():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               detection_nms_threshold = 0.5,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3(training=True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               detection_nms_threshold = 0.3,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'rots' : True })

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset()
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DSB2018_Dataset()
        dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
        dataset_val.prepare()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
        
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=50,
                    layers='all')
    else:
        dataset_test = DSB2018_Dataset()
        dataset_test.add_nuclei(_config.test_data_root, 'test')
        dataset_test.prepare()
        return _config, dataset_test


def train_resnet101_flips_alldata_minimask12_double_invert():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset(invert_type = 2)
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset(invert_type = 2)
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask12_double_invert_detectionnms0_2():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               detection_nms_threshold = 0.2,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset_train = DSB2018_Dataset(invert_type = 2)
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset(invert_type = 2)
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train_resnet101_flips_alldata_minimask12_double_invert_aug2():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = '2inv',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               fn_load = 'load_image_gt_augment_2')

    # Training dataset
    dataset_train = DSB2018_Dataset(invert_type = 2)
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset(invert_type = 2)
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=30,
                layers='all')


def train():

    _config = mask_rcnn_config(augmentation_dict = {'dim_ordering': 'tf', 
                                                    'fill_mode': 'reflect',
                                                    'horizontal_flip': True, 
                                                    'vertical_flip': True, 
                                                    'rotation_range': 45, 
                                                    'shear_range': 0.3,
                                                    'zoom_range': [0.9, 1]})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.9)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.9)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE / 10,
                epochs=50,
                layers='all')


def train_supplementaryval():

    _config = mask_rcnn_config(val_data_root = supplementary_dir)

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1.0)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.0)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=100,
                layers='all')


def trainsupplementary_val_trainsupplementary():

    _config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir)

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.8)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.8)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=100,
                layers='all')


def trainsupplementary_val_trainsupplementary_res50_imagenet():

    _config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               init_with = 'imagenet',
                               architecture = 'resnet50',
                               augmentation_dict = {'dim_ordering': 'tf', 
                                                    'fill_mode': 'reflect',
                                                    'horizontal_flip': True, 
                                                    'vertical_flip': True, 
                                                    'rotation_range': 45, 
                                                    'shear_range': 0.3,
                                                    'zoom_range': [0.9, 1]})

    # Training dataset
    dataset_train = DSB2018_Dataset()
    dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.8)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DSB2018_Dataset()
    dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.8)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=_config,
                              model_dir=_config.MODEL_DIR)
    model = load_weights(model, _config)
    
    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    model.train(dataset_train, dataset_val,
                learning_rate=_config.LEARNING_RATE / 10,
                epochs=30,
                layers='all')


def main():
    train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3()

if __name__ == '__main__':
    main()