import os
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
np.random.seed(1234)
import model as modellib
from model import log
import utils
import random
from settings import base_dir, train_dir, test_dir, ISBI_dir, nsb_dir, GAN_dir, stage2_test_dir
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


def train_resnet101_flipsrot_minimask12_double_invert_semantic(training = True):

    _config = mask_rcnn_config(train_data_root = [train_dir] + [test_dir] + [ISBI_dir] + [nsb_dir] + [GAN_dir],
                               test_data_root = [stage2_test_dir],
                               init_with = 'coco',
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
        dataset_train.add_nuclei(_config.train_data_root, 'train', 1.)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = None

        # Create model in training mode
        model = getattr(modellib, model_name)(mode="training", config=_config,
                                  model_dir=_config.MODEL_DIR)
        model = load_weights(model, _config)
    
        model.train(dataset_train, dataset_val,
                    learning_rate=_config.LEARNING_RATE,
                    epochs=25,
                    layers='all')
    else:
        dataset = DSB2018_Dataset(invert_type = 2)
        dataset.add_nuclei(test_dir, 'stage2_test', shuffle = False)
        dataset.prepare()
        return _config, dataset, model_name


def train_resnet101_flipsrot_minimask12_double_invert_semantic_b_w_colour(training = True):

    bw_config = mask_rcnn_config(train_data_root = [train_dir] + [test_dir] + [ISBI_dir] + [nsb_dir] + [GAN_dir],
                    test_data_root = [stage2_test_dir],
                    init_with = 'coco',
                    architecture = 'resnet101',
                    mini_mask_shape = 12,
                    images_per_gpu = 2, 
                    identifier = 'double_invert_semantic_bw',
                    augmentation_dict = {'dim_ordering': 'tf',
                                        'horizontal_flip': True,
                                        'vertical_flip': True, 
                                        'rots' : True })


    colour_config = mask_rcnn_config(train_data_root = [train_dir] + [test_dir] + [ISBI_dir] + [nsb_dir] + [GAN_dir],
                        test_data_root = [stage2_test_dir],
                        init_with = 'coco',
                        architecture = 'resnet101',
                        mini_mask_shape = 12,
                        images_per_gpu = 2, 
                        identifier = 'double_invert_semantic_colour',
                        augmentation_dict = {'dim_ordering': 'tf',
                                            'horizontal_flip': True,
                                            'vertical_flip': True, 
                                            'rots' : True })

    model_name = 'BespokeMaskRCNN'

    if training:
            ################
            # b/w model 
            ################

            # Training dataset
            dataset_train = DSB2018_Dataset(invert_type = 2)
            dataset_train.add_nuclei(_config.train_data_root, 'train', split_ratio = 1., target_colour_id = np.array([1]))
            dataset_train.prepare()

            # Validation dataset
            dataset_val = None

            # Create model in training mode
            bw_model = getattr(modellib, model_name)(mode="training", config=bw_config,
                                      model_dir=bw_config.MODEL_DIR)
            bw_model = load_weights(bw_model, bw_config)
    
            bw_model.train(dataset_train, dataset_val,
                        learning_rate=_config.LEARNING_RATE,
                        epochs=25,
                        layers='all')

            ################
            # colour model 
            ################

            # Training dataset
            dataset_train = DSB2018_Dataset(invert_type = 2)
            dataset_train.add_nuclei(colour_config.train_data_root, 'train', split_ratio = 1., target_colour_id = np.array([2]))
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
                        learning_rate=_config.LEARNING_RATE,
                        epochs=10,
                        layers='all')

    else:

        bw_dataset = DSB2018_Dataset(invert_type = 2)
        bw_dataset.add_nuclei(stage2_test_dir, 'stage2_test', shuffle = False, target_colour_id = np.array([1]))
        bw_dataset.prepare()

        colour_dataset = DSB2018_Dataset(invert_type = 2)
        colour_dataset.add_nuclei(stage2_test_dir, 'stage2_test', shuffle = False, target_colour_id = np.array([2]))
        colour_dataset.prepare()

        return [bw_config, colour_config], [bw_dataset, colour_dataset], model_name


def main():
    train_resnet101_flipsrot_minimask12_double_invert_semantic()
    train_resnet101_flipsrot_minimask12_double_invert_semantic_b_w_colour()

if __name__ == '__main__':
    main()