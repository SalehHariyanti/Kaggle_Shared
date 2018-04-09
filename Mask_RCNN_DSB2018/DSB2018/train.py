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
USER = getpass.getuser()

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
    model_name = 'MaskRCNN'

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
        model = getattr(modellib, model_name)(mode="training", config=_config,
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
        return _config, dataset, model_name


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

def train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3_nocache_color_balanced_safe(training=True):
    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'flips_rots_color_balanced',
                               augmentation_crop = 0,#0.5,
                               fn_load = 'load_image_gt_augment',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'zoom_range': (0.5,0.5),
                                                    'height_shift_range': 0.6,
                                                    'width_shift_range': 0,
                                                    'safe_transform' : True,
                                                    'rots':True})

    dataset_kwargs = { 'to_grayscale' : False , 'cache' : DSB2018_Dataset.Cache.NONE }

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1)
        dataset_train.prepare()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=80,
                    layers='all',
                    show_image_each = 100,
                    balance_by_cluster_id = True)

    else:

        dataset_test = DSB2018_Dataset(**dataset_kwargs)
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


def train_resnet101_flips_alldata_minimask12_double_invert_mosaics_plus_orig(training=True):

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
        configs = [_config] * 2

        # Mosaics dataset:
        dataset1 = DSB2018_Dataset(invert_type = 2)
        dataset1.add_nuclei(test_mosaics_dir, 'test', shuffle = False, use_mosaics=True)
        dataset1.prepare()
        # Original dataset
        dataset2 = DSB2018_Dataset(invert_type = 2)
        dataset2.add_nuclei(test_dir, 'test', shuffle = False)
        dataset2.prepare()

        dataset = [dataset1, dataset2]

        return configs, dataset


def train_resnet101_flipsrot_minimask12_double_invert_semantic_trainsupp(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               images_per_gpu = 2, 
                               identifier = 'double_invert_semantic',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots' : True })
    model_name = 'BespokeMaskRCNN'

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        for repeats in range(664//36):
          dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(invert_type = 2)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=30,
                    layers='all')
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name

def train_resnet101_flips_256_minimask12_double_invert(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               image_max_dim = 256,
                               image_min_dim = 256,
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
                    epochs=20,
                    layers='all')
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset

def train_resnet101_flipsrotzoom_alldata_minimask12_double_invert_semantic(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               images_per_gpu = 2,
                               identifier = 'double_invert',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'zoom_range': [0.8, 1],
                                                    'rots': True,
                                                    })
    model_name = 'BespokeMaskRCNN'

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
        model = getattr(modellib, model_name)(mode="training", config=_config,
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
        return _config, dataset, model_name

def train_resnet101_flipsrotzoom_minimask56_double_invert_semantic(training = True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 56,
                               images_per_gpu = 2, 
                               identifier = 'double_invert_semantic',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'zoom_range': [0.8, 1],
                                                    'rots' : True })
    model_name = 'BespokeMaskRCNN'

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(invert_type = 2)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=25,
                    layers='all')
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def train_resnet101_flipsrots_minimask12_nsbval(training=True):

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               train_data_root = train_dir,
                               val_data_root = [os.path.join(base_dir, 'train_external', 'nsb'), 
                                                os.path.join(base_dir, 'train_external', 'nsb_crop'), 
                                                os.path.join(base_dir, 'train_external', 'ISBI')],
                               mini_mask_shape = 12,
                               identifier = 'suppval',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots':True})

    if training:
        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1.0)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = DSB2018_Dataset(invert_type = 2)
        dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.0)
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
        dataset.add_nuclei([os.path.join(base_dir, 'train_external', 'nsb'), 
                            os.path.join(base_dir, 'train_external', 'nsb_crop'), 
                            os.path.join(base_dir, 'train_external', 'ISBI')], 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset


def trainsupp_resnet101_flipsrot_minimask12_double_invert_semantic_config2(training = True):

    model_name = 'BespokeMaskRCNN'
    dataset_kwargs = { 'invert_type' : 2 , 'cache' : DSB2018_Dataset.Cache.NONE }

    if training:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.9,
                            identifier = 'double_invert_semantic',
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        for repeats in range(664//36):
          dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(**dataset_kwargs)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=30,
                    layers='all')
    else:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.7,
                            identifier = 'double_invert_semantic',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        dataset = DSB2018_Dataset(**dataset_kwargs)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name

def trainsupp_resnet50_flipsrot_minimask12_double_invert_semantic_config2(training = True):

    model_name = 'BespokeMaskRCNN'
    dataset_kwargs = { 'invert_type' : 2 , 'cache' : DSB2018_Dataset.Cache.NONE }

    if training:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet50',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.9,
                            identifier = 'double_invert_semantic',
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        for repeats in range(664//36):
          dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(**dataset_kwargs)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=100,
                    layers='all')
    else:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet50',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.7,
                            identifier = 'double_invert_semantic',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        dataset = DSB2018_Dataset(**dataset_kwargs)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name

def trainsupp_resnet101_flipsrot_minimask12_no_invert_semantic_config2(training = True):

    model_name = 'BespokeMaskRCNN'
    dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.NONE }

    if training:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.9,
                            identifier = 'no_invert_semantic',
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        for repeats in range(664//36):
          dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(**dataset_kwargs)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=100,
                    layers='all')
    else:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.7,
                            identifier = 'no_invert_semantic',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        dataset = DSB2018_Dataset(**dataset_kwargs)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def train_resnet101_flipsrots_minimask12_nsbval(training=True):

   _config = mask_rcnn_config(init_with = 'coco',
                              architecture = 'resnet101',
                              train_data_root = train_dir,
                              val_data_root = [os.path.join(base_dir, 'train_external', 'nsb'), os.path.join(base_dir, 'train_external', 'nsb_crop'), os.path.join(base_dir, 'train_external', 'ISBI')],
                              mini_mask_shape = 12,
                              identifier = 'suppval',
                              augmentation_dict = {'dim_ordering': 'tf',
                                                   'horizontal_flip': True,
                                                   'vertical_flip': True,
                                                   'rots':True})
   model_name = 'MaskRCNN'

   # Validation dataset
   dataset_val = DSB2018_Dataset(invert_type = 2)
   dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.0)
   dataset_val.prepare()

   if training:
       # Training dataset
       dataset_train = DSB2018_Dataset(invert_type = 2)
       dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1.0)
       dataset_train.prepare()

       # Validation dataset
       #dataset_val = DSB2018_Dataset(invert_type = 2)
       #dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.0)
       #dataset_val.prepare()

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
       dataset.add_nuclei(test_dir, 'test', shuffle = False)
       dataset.prepare()
       return _config, dataset_val, model_name

def train_resnet101_flipsrot_minimask12_double_invert_semantic_config2(training = True):


    model_name = 'BespokeMaskRCNN'

    if training:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.9,
                            identifier = 'double_invert_semantic',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        # Training dataset
        dataset_train = DSB2018_Dataset(invert_type = 2)
        dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 0.995 if USER != 'antor' else 1.)
        dataset_train.prepare()

        if USER != 'antor':
            # Validation dataset
            dataset_val = DSB2018_Dataset(invert_type = 2)
            dataset_val.add_nuclei(_config.val_data_root, 'val', split_ratio = 0.995)
            dataset_val.prepare()

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, None if USER == 'antor' else dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=30,
                    layers='all')
    else:
        _config = mask_rcnn_config2(init_with = 'coco',
                            architecture = 'resnet101',
                            mini_mask_shape = 12,
                            images_per_gpu = 2, 
                            rpn_nms_threshold = 0.7,
                            identifier = 'double_invert_semantic',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})

        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def main():
    #train_resnet101_flips_alldata_minimask12_double_invert()
    if USER == 'antor':
        #train_resnet101_flipsrot_minimask12_double_invert_semantic()
        train_resnet101_flipsrot_minimask12_double_invert_semantic_config2()
    else:
        train_resnet101_flipsrot_minimask12_no_invert_semantic_config2()

if __name__ == '__main__':
    main()