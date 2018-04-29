import os
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
np.random.seed(1234)
import model as modellib
from model import log
import utils
import random
from settings import base_dir, train_dir, test_dir, supplementary_dir, stage2_test_dir
import getpass
USER = getpass.getuser()


TESTING  = False

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


def load_weights_from_model(model, weights_model, epoch = None):

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if epoch is not None:
        model_path = weights_model.find_last()[1][:-7] + '00' + str(epoch) + '.h5'
    else:
        model_path = weights_model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    return model


def train_resnet_semantic(training = True, architecture = 'resnet101'):
    """
    Generalised model fit on train/test/supplementary data:
    - semantic included
    - greyscale (no inverting)
    - original config, with max_gt_instances = 400
    - fn_load = load_image_gt_augment_nsb
    - dataset add_nuclei() adjustment so that supplementary images are equally represented
    - rpn_nms_threshold 0.9 in training, 0.7 when submitting
    """

    model_name = 'SemanticMaskRCNN'
    dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.DISK }
    identifier = 'semantic'
    identifier = '_'.join((identifier, 'res101' if architecture == 'resnet101' else 'res50'))

    # Training config
    _config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                        test_data_root = [stage2_test_dir],
                        init_with = 'coco',
                        architecture = architecture,
                        mini_mask_shape = 12,
                        max_gt_instances = 400,
                        rpn_nms_threshold = 0.9,
                        images_per_gpu = 2, 
                        identifier = identifier,
                        fn_load = 'load_image_gt_augment_nsb',
                        augmentation_dict = {'dim_ordering': 'tf',
                                            'horizontal_flip': True,
                                            'vertical_flip': True, 
                                            'rots' : True,
                                            'gaussian_blur': [-0.2, 0.2]})

    if training:

        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(train_dir, 'train', split_ratio = 1.)
        dataset_train.add_nuclei(test_dir, 'train', split_ratio = 1.)
        for repeats in range(730//36):
          dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 1.)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=1 if TESTING else 50,
                    layers='all',
                    augment_val = True)

    else:

        _config.RPN_NMS_THRESHOLD = 0.7

        dataset = DSB2018_Dataset(**dataset_kwargs)
        dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def train_resnet_semantic_maskcount_balanced(training = True, architecture = 'resnet101'):
    """
    Generalised model fit on train/test/supplementary data:
    - semantic included
    - greyscale (no inverting)
    - original config, with max_gt_instances = 400
    - fn_load = load_image_gt_augment_nsb
    - dataset add_nuclei() adjustment so that supplementary images are equally represented
    - rpn_nms_threshold 0.9 in training, 0.7 when submitting
    """

    model_name = 'SemanticMaskRCNN'
    dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.DISK }
    identifier = 'semantic_bal'
    identifier = '_'.join((identifier, 'res101' if architecture == 'resnet101' else 'res50'))

    # Training config
    _config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                        test_data_root = [stage2_test_dir],
                        init_with = 'coco',
                        architecture = architecture,
                        mini_mask_shape = 12,
                        max_gt_instances = 400,
                        rpn_nms_threshold = 0.9,
                        images_per_gpu = 2, 
                        identifier = identifier,
                        fn_load = 'load_image_gt_augment_nsb',
                        augmentation_dict = {'dim_ordering': 'tf',
                                            'horizontal_flip': True,
                                            'vertical_flip': True, 
                                            'rots' : True,
                                            'gaussian_blur': [-0.2, 0.2]})

    if training:

        # Training dataset
        dataset_train = DSB2018_Dataset(**dataset_kwargs)
        dataset_train.add_nuclei(train_dir, 'train', split_ratio = 1.)
        dataset_train.add_nuclei(test_dir, 'train', split_ratio = 1.)
        for repeats in range(730 // 36):
            dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 1.)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=1 if TESTING else 50,
                    layers='all',
                    augment_val = True,
                    balance_by_cluster_id = True, str_cluster_id = 'maskcount_id')

    else:

        _config.RPN_NMS_THRESHOLD = 0.7

        dataset = DSB2018_Dataset(**dataset_kwargs)
        dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def train_resnet_semantic_b_w_colour(training = True, architecture='resnet101'):
    """
    b/w and colour models: 
    - semantic included
    - greyscale + invert_type == 2 / 0
    - original config, with max_gt_instances = 400
    - fn_load = b/w: load_image_gt_augment; colour: load_image_gt_augment_nsb
    - rpn_nms_threshold 0.9 in training, 0.7 when submitting
    """

    model_name = 'SemanticMaskRCNN'
    bw_dataset_kwargs = { 'invert_type' : 2 , 'cache' : DSB2018_Dataset.Cache.DISK }
    colour_dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.DISK }

    bw_identifier = 'semantic_bw'
    bw_identifier = '_'.join((bw_identifier, 'res101' if architecture == 'resnet101' else 'res50'))

    colour_identifier = 'semantic_colour'
    colour_identifier = '_'.join((colour_identifier, 'res101' if architecture == 'resnet101' else 'res50'))

    bw_config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                            test_data_root = [stage2_test_dir],
                            init_with = 'coco',
                            architecture = architecture,
                            mini_mask_shape = 12,
                            max_gt_instances = 400,
                            rpn_nms_threshold = 0.9,
                            images_per_gpu = 1, 
                            identifier = bw_identifier,
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})


    colour_config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                            test_data_root = [stage2_test_dir],
                            init_with = 'coco',
                            architecture = architecture,
                            mini_mask_shape = 12,
                            max_gt_instances = 400,
                            rpn_nms_threshold = 0.9,
                            images_per_gpu = 1, 
                            identifier = colour_identifier,
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})


    if training:
        ################
        # b/w model 
        ################

        # Training dataset
        dataset_train = DSB2018_Dataset(**bw_dataset_kwargs)
        dataset_train.add_nuclei(bw_config.train_data_root, 'train', split_ratio = 1., target_colour_id = np.array([1]))
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        bw_model = getattr(modellib, model_name)(mode="training", config=bw_config,
                                    model_dir=bw_config.MODEL_DIR)
        bw_model = load_weights(bw_model, bw_config)
            
        bw_model.train(dataset_train, dataset_val,
                    learning_rate=bw_config.LEARNING_RATE,
                    epochs=1 if TESTING else 20,
                    layers='all')
            
        ################
        # colour model 
        ################

        # Training dataset
        dataset_train = DSB2018_Dataset(**colour_dataset_kwargs)
        dataset_train.add_nuclei(train_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        dataset_train.add_nuclei(test_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        for repeats in range(135//45):
            dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        colour_model = getattr(modellib, model_name)(mode="training", config=colour_config,
                                    model_dir=colour_config.MODEL_DIR)
        colour_model = load_weights_from_model(colour_model, bw_model)
    
        # Clear bw_model
        bw_model = None

        colour_model.train(dataset_train, dataset_val,
                    learning_rate=colour_config.LEARNING_RATE,
                    epochs=1 if TESTING else 10,
                    layers='all',
                    augment_val = True)

    else:

        bw_dataset = DSB2018_Dataset(**bw_dataset_kwargs)
        bw_dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False, target_colour_id = np.array([1]))
        bw_dataset.prepare()

        colour_dataset = DSB2018_Dataset(**colour_dataset_kwargs)
        colour_dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False, target_colour_id = np.array([2]))
        colour_dataset.prepare()

        bw_config.RPN_NMS_THRESHOLD = 0.7
        colour_config.RPN_NMS_THRESHOLD = 0.7

        return [bw_config, colour_config], [bw_dataset, colour_dataset], [model_name, model_name]


def train_resnet_semantic_b_w_colour_maskcount_balanced(training = True, architecture='resnet101'):
    """
    b/w and colour models: 
    - semantic included
    - greyscale + invert_type == 2 / 0
    - original config, with max_gt_instances = 400
    - fn_load = b/w: load_image_gt_augment; colour: load_image_gt_augment_nsb
    - rpn_nms_threshold 0.9 in training, 0.7 when submitting
    """

    model_name = 'SemanticMaskRCNN'
    bw_dataset_kwargs = { 'invert_type' : 2 , 'cache' : DSB2018_Dataset.Cache.DISK }
    colour_dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.DISK }

    bw_identifier = 'semantic_bw_bal'
    bw_identifier = '_'.join((bw_identifier, 'res101' if architecture == 'resnet101' else 'res50'))

    colour_identifier = 'semantic_colour_bal'
    colour_identifier = '_'.join((colour_identifier, 'res101' if architecture == 'resnet101' else 'res50'))


    bw_config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                            test_data_root = [stage2_test_dir],
                            init_with = 'coco',
                            architecture = architecture,
                            mini_mask_shape = 12,
                            max_gt_instances = 400,
                            rpn_nms_threshold = 0.9,
                            images_per_gpu = 1, 
                            identifier = bw_identifier,
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})


    colour_config = mask_rcnn_config(train_data_root = [train_dir] + supplementary_dir + [test_dir],
                            test_data_root = [stage2_test_dir],
                            init_with = 'coco',
                            architecture = architecture,
                            mini_mask_shape = 12,
                            max_gt_instances = 400,
                            rpn_nms_threshold = 0.9,
                            images_per_gpu = 1, 
                            identifier = colour_identifier,
                            fn_load = 'load_image_gt_augment_nsb',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True, 
                                                'rots' : True,
                                                'gaussian_blur': [-0.2, 0.2]})


    if training:
        ################
        # b/w model 
        ################

        # Training dataset
        dataset_train = DSB2018_Dataset(**bw_dataset_kwargs)
        dataset_train.add_nuclei(bw_config.train_data_root, 'train', split_ratio = 1., target_colour_id = np.array([1]))
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        bw_model = getattr(modellib, model_name)(mode="training", config=bw_config,
                                    model_dir=bw_config.MODEL_DIR)
        bw_model = load_weights(bw_model, bw_config)
            
        bw_model.train(dataset_train, dataset_val,
                    learning_rate=bw_config.LEARNING_RATE,
                    epochs=1 if TESTING else 20,
                    layers='all',
                    balance_by_cluster_id = True, str_cluster_id = 'maskcount_id')
            
        ################
        # colour model 
        ################

        # Training dataset
        dataset_train = DSB2018_Dataset(**colour_dataset_kwargs)
        dataset_train.add_nuclei(train_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        dataset_train.add_nuclei(test_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        for repeats in range(135//45):
            dataset_train.add_nuclei(supplementary_dir, 'train', split_ratio = 1., target_colour_id = np.array([2]))
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        colour_model = getattr(modellib, model_name)(mode="training", config=colour_config,
                                    model_dir=colour_config.MODEL_DIR)
        colour_model = load_weights_from_model(colour_model, bw_model)
    
        # Clear bw_model
        bw_model = None

        colour_model.train(dataset_train, dataset_val,
                    learning_rate=colour_config.LEARNING_RATE,
                    epochs=1 if TESTING else 10,
                    layers='all',
                    augment_val = True,
                    balance_by_cluster_id = True, str_cluster_id = 'maskcount_id')

    else:

        bw_dataset = DSB2018_Dataset(**bw_dataset_kwargs)
        bw_dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False, target_colour_id = np.array([1]))
        bw_dataset.prepare()

        colour_dataset = DSB2018_Dataset(**colour_dataset_kwargs)
        colour_dataset.add_nuclei(stage2_test_dir, 'test', shuffle = False, target_colour_id = np.array([2]))
        colour_dataset.prepare()

        bw_config.RPN_NMS_THRESHOLD = 0.7
        colour_config.RPN_NMS_THRESHOLD = 0.7

        return [bw_config, colour_config], [bw_dataset, colour_dataset], [model_name, model_name]


def train_resnet101_semantic(training=True):
  return train_resnet_semantic(training=training, architecture='resnet101')

def train_resnet50_semantic(training=True):
  return train_resnet_semantic(training=training, architecture='resnet50')

def train_resnet101_semantic_maskcount_balanced(training=True):
  return train_resnet_semantic_maskcount_balanced(training=training, architecture='resnet101')

def train_resnet50_semantic_maskcount_balanced(training=True):
  return train_resnet_semantic_maskcount_balanced(training=training, architecture='resnet50')

def train_resnet101_semantic_b_w_colour(training=True):
  return train_resnet_semantic_b_w_colour(training=training, architecture='resnet101')

def train_resnet50_semantic_b_w_colour(training=True):
  return train_resnet_semantic_b_w_colour(training=training, architecture='resnet50')

def train_resnet101_semantic_b_w_colour_maskcount_balanced(training=True):
  return train_resnet_semantic_b_w_colour_maskcount_balanced(training=training, architecture='resnet101')

def train_resnet50_semantic_b_w_colour_maskcount_balanced(training=True):
  return train_resnet_semantic_b_w_colour_maskcount_balanced(training=training, architecture='resnet50')


def train_resnet_semantic_kfold(training = True, architecture = 'resnet101', k = 5):
    """
    Generalised model fit on train/test/supplementary data:
    - semantic included
    - greyscale (no inverting)
    - original config, with max_gt_instances = 400
    - fn_load = load_image_gt_augment_nsb
    - dataset add_nuclei() adjustment so that supplementary images are equally represented
    - rpn_nms_threshold 0.9 in training, 0.7 when submitting
    """

    dataset_kwargs = { 'invert_type' : 0 , 'cache' : DSB2018_Dataset.Cache.DISK }
            
    if training:

        for _k in range(k):

            model_name = 'SemanticMaskRCNN'
            identifier = '_'.join(('semantic', 'kfold', str(_k), str(k)))
            identifier = '_'.join((identifier, 'res101' if architecture == 'resnet101' else 'res50'))

            # Training config
            _config = mask_rcnn_config(train_data_root = [train_dir] + [test_dir],
                                test_data_root = [stage2_test_dir],
                                init_with = 'coco',
                                architecture = architecture,
                                mini_mask_shape = 12,
                                max_gt_instances = 400,
                                rpn_nms_threshold = 0.9,
                                images_per_gpu = 1, 
                                identifier = identifier,
                                fn_load = 'load_image_gt_augment_nsb',
                                augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots' : True,
                                                    'gaussian_blur': [-0.2, 0.2]})

            # Training dataset
            dataset_train = DSB2018_Dataset(**dataset_kwargs)
            dataset_train.add_nuclei(train_dir, 'train', split_ratio = (1 - 1 / k), kfold = _k)
            dataset_train.add_nuclei(test_dir, 'train', split_ratio = (1 - 1 / k), kfold = _k)
            dataset_train.prepare()

            # Validation dataset
            dataset_val = None

            # Create model in training mode
            model = getattr(modellib, model_name)(mode="training", config=_config,
                                        model_dir=_config.MODEL_DIR)
            model = load_weights(model, _config)
    
            model.train(dataset_train, dataset_val,
                        learning_rate=_config.LEARNING_RATE,
                        epochs=1 if TESTING else 20,
                        layers='all',
                        augment_val = True)


    else:

        model_name = 'SemanticMaskRCNN'
            
        dataset, config = [], []

        for _k in range(k):
            dataset_train = DSB2018_Dataset(**dataset_kwargs)
            dataset_train.add_nuclei(stage2_test_dir, 'test', shuffle = False, split_ratio = 1.)
            dataset_train.prepare()
            dataset.append(dataset_train)

            identifier = '_'.join(('semantic', 'kfold', str(_k), str(k)))
            identifier = '_'.join((identifier, 'res101' if architecture == 'resnet101' else 'res50'))

            # Training config
            _config = mask_rcnn_config(train_data_root = [train_dir] + [test_dir],
                                test_data_root = [stage2_test_dir],
                                init_with = 'coco',
                                architecture = architecture,
                                mini_mask_shape = 12,
                                max_gt_instances = 400,
                                rpn_nms_threshold = 0.9,
                                images_per_gpu = 1, 
                                identifier = identifier,
                                fn_load = 'load_image_gt_augment_nsb',
                                augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True, 
                                                    'rots' : True,
                                                    'gaussian_blur': [-0.2, 0.2]})

            _config.RPN_NMS_THRESHOLD = 0.7

            config.append(_config)

        return config, dataset, [model_name] * k



def main():
    
    # train_resnet_semantic_kfold()

    train_resnet101_semantic()
    train_resnet50_semantic()
    train_resnet101_semantic_maskcount_balanced()
    train_resnet50_semantic_maskcount_balanced()
    train_resnet101_semantic_b_w_colour()
    train_resnet50_semantic_b_w_colour()
    train_resnet101_semantic_b_w_colour_maskcount_balanced()
    train_resnet50_semantic_b_w_colour_maskcount_balanced()



if __name__ == '__main__':
    main()