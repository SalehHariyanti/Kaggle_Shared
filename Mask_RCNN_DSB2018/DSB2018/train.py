import os
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
np.random.seed(1234)
import model as modellib
from model import log
import utils
import random
from settings import train_dir, supplementary_dir, train_mosaics_dir, test_mosaics_dir, base_dir
import getpass

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


def train_resnet101_flips_alldata_minimask12_double_invert(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    if training:
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
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset_test


def train_resnet101_flips_alldata_minimask12_double_invert_masksizes(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = '2inv',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               mask_size_dir = os.path.join(data_dir, 'mask_sizes'))

    if training:
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

    else:

        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset


def train_resnet101_flips_alldata_minimask12_double_invert_scaled(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = '2inv',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               fn_load = 'load_image_gt_augment_scaled',
                               mask_size_dir = os.path.join(data_dir, 'mask_sizes'))

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1.0)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=30,
                    layers='all')
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset


def train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3_mosaics(training=True):
    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               train_data_root = train_mosaics_dir,
                               val_data_root = train_mosaics_dir,
                               test_data_root = test_mosaics_dir,
                               mini_mask_shape = 12,
                               identifier = 'double_invert_mosaics',
                               augmentation_crop = 1.,
                               fn_load = 'load_image_gt_augment',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots':True})

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1, use_mosaics=True)
        dataset_train.prepare()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=20,
                    layers='all',
                    show_image_each = 100)

    else:

        dataset_test = DSB2018_Dataset(invert_type = 2)
        dataset_test.add_nuclei(test_dir, 'test', shuffle = False)
        dataset_test.prepare()
        return _config, dataset_test

def train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3_nocache_color(training=True):
    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert_mosaics',
                               augmentation_crop = 1.,
                               fn_load = 'load_image_gt_augment',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots':True})

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(to_grayscale = False, cache = DSB2018_Dataset.Cache.NONE)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1)
        dataset_train.prepare()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=20,
                    layers='all',
                    show_image_each = 100)

    else:

        dataset_test = DSB2018_Dataset(invert_type = 2)
        dataset_test.add_nuclei(test_dir, 'test', shuffle = False)
        dataset_test.prepare()
        return _config, dataset_test

def train_resnet101_flips_all_rots_data_minimask12_mosaics_nsbval(training=True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               train_data_root = train_mosaics_dir,
                               val_data_root = [os.path.join(base_dir, 'train_external', 'nsb'), os.path.join(base_dir, 'train_external', 'ISBI')],
                               mini_mask_shape = 12,
                               identifier = '2inv_mos',
                               val_steps = 45,
                               augmentation_crop = 1.,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots':True})

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1.0, use_mosaics=True)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DSB2018_Dataset(invert_type = 2)
        dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.0, use_mosaics=False)
        dataset_val.prepare()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=30,
                    layers='all',
                    augment_val = True)

    else:

        dataset = DSB2018_Dataset(invert_type = 2)
        #dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.add_nuclei(test_mosaics_dir, 'test', shuffle = False, use_mosaics = True)
        dataset.prepare()
        return _config, dataset


def main():
    #train_resnet101_flips_alldata_minimask12_double_invert()
    if getpass.getuser() == 'antor':
        train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3_nocache_color()
    else:
        #train_resnet101_flips_alldata_minimask12_double_invert_scaled()
        #train_resnet101_flips_all_rots_data_minimask12_mosaics_nsbval()
        train_resnet101_flips_alldata_minimask12_double_invert()

if __name__ == '__main__':
    main()