import os
import datetime
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
import model as modellib
import functions as f
from settings import test_dir, submissions_dir, supplementary_dir
import utils
import dsb2018_utils as du
import scipy
import cv2

import visualize
import matplotlib.pyplot as plt
                    
def base_test_dataset():
    dataset = DSB2018_Dataset()
    dataset.add_nuclei(test_dir, 'test')
    dataset.prepare()
    return dataset


def maskrcnn_detect_flips(model, images, threshold):
    """
    Rotates images by 4 x 90 degrees and combines results via non-maximum suppression
    """

    images_flip = [images, 
                    [np.fliplr(img) for img in images], 
                    [np.flipud(img) for img in images],
                    [np.flipud(np.fliplr(img)) for img in images],
                    [np.fliplr(np.flipud(img)) for img in images]]

    results_flip = [model.detect(img, verbose=0) for img in images_flip]

    # Reverse flip masks
    for i in range(len(images)):
        results_flip[1][i]['masks'] = np.fliplr(results_flip[1][i]['masks'])
        results_flip[2][i]['masks'] = np.flipud(results_flip[2][i]['masks'])
        results_flip[3][i]['masks'] = np.fliplr(np.flipud(results_flip[3][i]['masks']))
        results_flip[4][i]['masks'] = np.flipud(np.fliplr(results_flip[4][i]['masks']))
        # Recalculate bboxes
        for j in range(len(results_flip)):
            results_flip[j][i]['rois'] = utils.extract_bboxes(results_flip[j][i]['masks'])
            # Reshape masks so that they can be concatenated correctly
            results_flip[j][i]['masks'] = np.moveaxis(results_flip[j][i]['masks'], -1, 0)

    # Carry out non-maximum suppression to reduce results_flip for each image to a single set of results
    results = []
    for i in range(len(images)):

        img_results_flip = du.concatenate_list_of_dicts([rf[i] for rf in results_flip])
        nms_idx = utils.non_max_suppression(img_results_flip['rois'], img_results_flip['scores'].reshape(-1, ), threshold)
        img_results_flip = du.reduce_dict(img_results_flip, nms_idx)

        # Reshape masks
        img_results_flip['masks'] = np.moveaxis(img_results_flip['masks'], 0, -1)
        img_results_flip['class_ids'] = img_results_flip['class_ids'].reshape(-1, )
        img_results_flip['scores'] = img_results_flip['scores'].reshape(-1, )

        results.append(img_results_flip)

    return results


def maskrcnn_detect_scale(model, images, threshold, scales = [0.6, 0.8, 1.2, 1.4]):
    """
    Rescales images to specified scales and combines results via non-maximum suppression
    """

    images_scale = [images] + [[scipy.misc.imresize(img, (round(img.shape[0] * scale), round(img.shape[1] * scale))) for img in images] for scale in scales]


    results_scale = [model.detect(img, verbose=0) for img in images_scale]

    # Reverse the scales
    for i in range(len(images)):
        for j in range(len(results_scale)):
            results_scale[j][i]['masks'] = scipy.ndimage.zoom(results_scale[j][i]['masks'], 
                                                              (images[i].shape[0] / results_scale[j][i]['masks'].shape[0], 
                                                               images[i].shape[1] / results_scale[j][i]['masks'].shape[1], 
                                                               1), order = 0)
            results_scale[j][i]['rois'] = utils.extract_bboxes(results_scale[j][i]['masks'])
            # Reshape masks so that they can be concatenated correctly
            results_scale[j][i]['masks'] = np.moveaxis(results_scale[j][i]['masks'], -1, 0)

    # Carry out non-maximum suppression to reduce results_flip for each image to a single set of results
    results = []
    for i in range(len(images)):

        img_results_scale = du.concatenate_list_of_dicts([rf[i] for rf in results_scale])
        nms_idx = utils.non_max_suppression(img_results_scale['rois'], img_results_scale['scores'].reshape(-1, ), threshold)
        img_results_scale = du.reduce_dict(img_results_scale, nms_idx)

        # Reshape masks
        img_results_scale['masks'] = np.moveaxis(img_results_scale['masks'], 0, -1)
        img_results_scale['class_ids'] = img_results_scale['class_ids'].reshape(-1, )
        img_results_scale['scores'] = img_results_scale['scores'].reshape(-1, )

        results.append(img_results_scale)

    return results


def create_model(_config, epoch = None):
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                                config=_config,
                                model_dir=_config.MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if epoch is not None:
        model_path = model.find_last()[1][:-7] + '00' + str(epoch) + '.h5'
    else:
        model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    return model


def load_dataset_images(dataset, i, batch_size):
    # Load image
    images = []
    for idx in range(i, i + batch_size):
        if idx < len(dataset.image_ids):
            images.append(dataset.load_image(dataset.image_ids[idx]))
        else:
            images.append(images[-1])
    return images


def predict_model(_config, dataset, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3, img_pad = 0, dilate = False):

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=_config,
                              model_dir=_config.MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    if epoch is not None:
        model_path = model.find_last()[1][:-7] + '00' + str(epoch) + '.h5'
    else:
        model_path = model.find_last()[1]

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    '''
    load test dataset one by one. Note that masks are resized (back) in model.detect
    rle2csv
    '''        
    ImageId = []
    EncodedPixels = []
    
    # NB: we need to predict in batches of _config.BATCH_SIZE
    # as there are layers within the model that have strides dependent on this.
    for i in range(0, len(dataset.image_ids), _config.BATCH_SIZE):
         # Load image
        images = []
        N = 0
        for idx in range(i, i + _config.BATCH_SIZE):
            if idx < len(dataset.image_ids):
                N += 1
                if img_pad > 0:
                    img = dataset.load_image(dataset.image_ids[idx])                  
                    images.append(np.stack([np.pad(img[:, :, i], img_pad, mode = 'reflect') for i in range(img.shape[-1])], axis = -1))
                else:
                    images.append(dataset.load_image(dataset.image_ids[idx]))
            else:
                images.append(images[-1])

        # Run detection
        if augment_flips:
            r = maskrcnn_detect_flips(model, images, threshold = nms_threshold)
        elif augment_scale:
            r = maskrcnn_detect_scale(model, images, threshold = nms_threshold)
        else:
            r = model.detect(images, verbose=0)
        
        # Reduce to N images
        for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

            if j < N:   

                masks = r[j]['masks'] #[H, W, N] instance binary masks

                if img_pad > 0:

                    masks = masks[img_pad : -img_pad, img_pad : -img_pad]
                    valid = np.sum(masks, axis = (0, 1)) > 0
                    masks = masks[:, :, valid]

                    r[j]['masks'] = masks
                    r[j]['scores'] = r[j]['scores'][valid]
                    r[j]['class_ids'] = r[j]['class_ids'][valid]
                    r[j]['rois'] = r[j]['rois'][valid]

    
                if dilate:
                    masks = np.stack([cv2.dilate(mask.astype(np.uint8), kernel = np.ones((3, 3), dtype = np.uint8)) for mask in np.moveaxis(masks, -1, 0)], axis = -1)

                img_name = dataset.image_info[idx]['name']
        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, r[j]['scores'])
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if False:
                    class_names = ['background', 'nucleus']
                    visualize.display_instances((images[j] * 255).astype(np.uint8), r[j]['rois'], r[j]['masks'], r[j]['class_ids'], class_names, r[j]['scores'], figsize = (8, 8))
                    

    f.write2csv(os.path.join(submissions_dir, '_'.join(('submission', _config.NAME, str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_multiple(configs, datasets, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3):

    ImageId = []
    EncodedPixels = []

    for _config, dataset in zip(configs, datasets):
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference", 
                                  config=_config,
                                  model_dir=_config.MODEL_DIR)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        if epoch is not None:
            model_path = model.find_last()[1][:-7] + '00' + str(epoch) + '.h5'
        else:
            model_path = model.find_last()[1]

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        '''
        load test dataset one by one. Note that masks are resized (back) in model.detect
        rle2csv
        '''        
    
        # NB: we need to predict in batches of _config.BATCH_SIZE
        # as there are layers within the model that have strides dependent on this.
        for i in range(0, len(dataset.image_ids), _config.BATCH_SIZE):
             # Load image
            images = []
            N = 0
            for idx in range(i, i + _config.BATCH_SIZE):
                if idx < len(dataset.image_ids):
                    N += 1
                    images.append(dataset.load_image(dataset.image_ids[idx]))
                else:
                    images.append(images[-1])

            # Run detection
            if augment_flips:
                r = maskrcnn_detect_flips(model, images, threshold = nms_threshold)
            elif augment_scale:
                r = maskrcnn_detect_scale(model, images, threshold = nms_threshold)
            else:
                r = model.detect(images, verbose=0)
        
            # Reduce to N images
            for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

                if j < N:   

                    masks = r[j]['masks'] #[H, W, N] instance binary masks
    
                    img_name = dataset.image_info[idx]['name']
        
                    ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, r[j]['scores'])
                    ImageId += ImageId_batch
                    EncodedPixels += EncodedPixels_batch

                    if False:
                        class_names = ['background', 'nucleus']
                        visualize.display_instances((images[j] * 255).astype(np.uint8), r[j]['rois'], r[j]['masks'], r[j]['class_ids'], class_names, r[j]['scores'], figsize = (8, 8))
                    

    f.write2csv(os.path.join(submissions_dir, '_'.join(('submission', 
                                                        '_'.join([_config.NAME for _config in configs]), 
                                                        str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_nms(configs, datasets, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3):

    ImageId = []
    EncodedPixels = []

    models = [create_model(_config, epoch) for _config in configs]
    n_images = len(datasets[0].image_ids)
    batch_size = max([_config.BATCH_SIZE for _config in configs])

    # NB: we need to predict in batches of _config.BATCH_SIZE
    # as there are layers within the model that have strides dependent on this.
    for i in range(0, n_images, batch_size):

        images = [load_dataset_images(dataset, i, batch_size) for dataset in datasets]

        # Run detection
        res = []
        for model, _images, _configs in zip(models, images, configs):
            if _configs.BATCH_SIZE == batch_size:
                res.append(model.detect(_images, verbose=0))
            else:
                assert _configs.BATCH_SIZE == 1
                res.append([model.detect([img], verbose=0)[0] for img in _images])
        
        # Reduce to N images
        for j, idx in enumerate(range(i, i + batch_size)):      

            if idx < n_images:   

                # Get masks via nms
                rois = np.concatenate([r[j]['rois'] for r in res])
                scores = np.concatenate([r[j]['scores'] for r in res])
                masks = np.moveaxis(np.concatenate([np.moveaxis(r[j]['masks'], -1, 0) for r in res]), 0, -1)

                nms_idx = utils.non_max_suppression(rois, scores, nms_threshold)

                masks = masks[:, :, nms_idx]
                scores = scores[nms_idx]
                rois = rois[nms_idx]
                
                img_name = datasets[0].image_info[idx]['name']
        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, scores)
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if False:
                    class_names = ['background', 'nucleus']
                    visualize.display_instances((images[j] * 255).astype(np.uint8), rois, masks, np.ones(scores.shape, dtype = np.int), class_names, scores, figsize = (8, 8))
                    

    f.write2csv(os.path.join(submissions_dir, '_'.join(('submission_nms', 
                                                    '_'.join([_config.NAME for _config in configs]), 
                                                    str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_1():

    _config = mask_rcnn_config()
    predict_model(_config, dataset = base_test_dataset(), epoch = None)


def predict_2():

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


    predict_model(_config, dataset = base_test_dataset())


def predict_3():

    _config = mask_rcnn_config(augmentation_dict = {'dim_ordering': 'tf', 
                                                'fill_mode': 'reflect',
                                                'horizontal_flip': True, 
                                                'vertical_flip': True, 
                                                'rotation_range': 45, 
                                                'shear_range': 0.3,
                                                'zoom_range': [0.9, 1]})

    predict_model(_config, dataset = base_test_dataset())


def predict_4():

    _config = mask_rcnn_config(augmentation_dict = {'dim_ordering': 'tf'})

    predict_model(_config, dataset = base_test_dataset())


def predict_5():
    # submission_DSB2018_512_512_dim_o-tf-horiz-True-verti-True_None_20180319182416_.csv
    # train_resnet101_flips
    _config = mask_rcnn_config(init_with = 'coco',
                            architecture = 'resnet101',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True})

    predict_model(_config)


def predict_6():
    # train_resnet101_flips, augmented detection with nms_threshold = 0.5
    # submission_DSB2018_512_512_dim_o-tf-horiz-True-verti-True_None_20180320125310_
    _config = mask_rcnn_config(init_with = 'coco',
                            architecture = 'resnet101',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True})

    predict_model(_config, dataset = base_test_dataset(), augment = True, nms_threshold = 0.5)


def predict_7():
    # train_resnet101_flips, augmented detection with nms_threshold = 0.3
    #submission_DSB2018_512_512_dim_o-tf-horiz-True-verti-True_None_20180320130000_
    _config = mask_rcnn_config(init_with = 'coco',
                            architecture = 'resnet101',
                            augmentation_dict = {'dim_ordering': 'tf',
                                                'horizontal_flip': True,
                                                'vertical_flip': True})

    predict_model(_config, dataset = base_test_dataset(), augment = True, nms_threshold = 0.3)


def predict_8():
   # train_resnet101_flips_by_colour_inc_suppdata
   #submission_DSB2018_512_512_colour_1_dim_o-tf-horiz-True-verti-True_DSB2018_512_512_colour_2_dim_o-tf-horiz-True-verti-True_None_20180322183515_

    config_grey = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_1')

    config_colour = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_2')


    dataset_grey = DSB2018_Dataset()
    dataset_grey.add_nuclei(test_dir, 'test', target_colour_id = np.array([1]))
    dataset_grey.prepare()

    dataset_colour = DSB2018_Dataset()
    dataset_colour.add_nuclei(test_dir, 'test', target_colour_id = np.array([2]))
    dataset_colour.prepare()

    datasets = [dataset_grey, dataset_colour]
    configs = [config_grey, config_colour]

    predict_multiple(configs, datasets) 


def predict_9():
    # submission_DSB2018_512_512__dim_o-tf-horiz-True-verti-True-y_gau--0.25_None_20180323085451_.csv
    # train_resnet101_flips_yblur():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'y_gaussian_blur': [-0.25, 0.25]})

    predict_model(_config, base_test_dataset())


def predict_10():
    # submission_DSB2018_512_512__dim_o-tf-horiz-True-verti-True_None_20180323141239_.csv
    # train_resnet101_flips_alldata():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_11():
    #submission_DSB2018_512_512__dim_o-tf-horiz-True-verti-True_0.8_None_20180323193035_
    # train_resnet101_flips_alldata_augcrop0p8():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               augmentation_crop = 0.8)

    predict_model(_config, base_test_dataset())


def predict_12():
    #submission_DSB2018_512_512_False_400__dim_o-tf-horiz-True-verti-True_0.5_None_20180324145831_
    # train_resnet101_flips_alldata_maxgt400_minimaskFalse():

    _config = mask_rcnn_config(init_with = 'coco',
                               max_gt_instances = 400,
                               use_mini_mask = False,
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_13():
    # def train_resnet101_flips_alldata():
    # but with cropping ranges of 30-90% of image size
    #submission_DSB2018_512_512_True_56_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180324213134_
    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})
    predict_model(_config, base_test_dataset())


def predict_14():
    #submission_DSB2018_512_512_True_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180325091000_
    #def train_resnet101_flips_alldata_minimask28():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 28,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_15():
    #submission_DSB2018_512_512_True_56_196__dim_o-tf-horiz-True-verti-True_0.5_None_20180325091259_
    #def train_resnet101_flips_alldata_maxgt196():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               max_gt_instances = 196,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_16():
    #submission_DSB2018_512_512_True_18_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180325223651_
    #def train_resnet101_flips_alldata_minimask18():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 18,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})
    predict_model(_config, base_test_dataset())


def predict_17():
    #submission_DSB2018_512_512_True_56_128__dim_o-tf-horiz-True-verti-True_0.5_None_20180325223949_
    #def train_resnet101_flips_alldata_maxgt128():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               max_gt_instances = 128,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})
    predict_model(_config, base_test_dataset())


def predict_18():
    #submission_DSB2018_512_512_True_12_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180326101255_
    # def train_resnet101_flips_alldata_minimask12():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())



def predict_19():
    #submission_DSB2018_512_512_True_8_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180326171446_
    #def train_resnet101_flips_alldata_minimask8():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 8,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())

def predict_20():
    #submission_DSB2018_512_512_True_12_28_196__dim_o-tf-horiz-True-verti-True_0.5_None_20180326215454_
    #def train_resnet101_flips_alldata_minimask12_maxgt196():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               max_gt_instances = 196,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_21():
    #submission_DSB2018_512_512_True_12_16_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180327140627_
    #def train_resnet101_flips_alldata_minimask12_mask16():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               mask_shape = 16,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_22():
    # submission_DSB2018_512_512_True_12_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180327143836_
    # def train_resnet101_flips_alldata_minimask12() with maskrcnn_detect_scale

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset(), augment_scale = True)


def predict_23():
    # submission_DSB2018_512_512_True_12_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180327151512_
    # def train_resnet101_flips_alldata_minimask12() with img_pad = 20

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset(), img_pad = 20)


def predict_24():
    #submission_DSB2018_512_512_True_12_28_256__contr-10-dim_o-tf-horiz-True-verti-True_0.5_None_20180328085021_
    #train_resnet101_flips_alldata_minimask12_augcontrast()

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True,
                                                    'contrast_stretching': 10})

    predict_model(_config, base_test_dataset())



def predict_25():
    #submission_DSB2018_512_512_True_12_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180328085322_
    #def train_resnet101_flips_alldata_minimask12_ep50():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_26():
    
   # train_resnet101_flips_by_colour_inc_suppdata
   #submission_nms_DSB2018_512_512_True_56_28_256_colour_1_dim_o-tf-horiz-True-verti-True_0.5_DSB2018_512_512_True_56_28_256_colour_2_dim_o-tf-horiz-True-verti-True_0.5_None_20180328125128_

    config_grey = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_1')

    config_colour = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True},
                               train_data_root = [train_dir] + supplementary_dir,
                               val_data_root = [train_dir] + supplementary_dir,
                               identifier = 'colour_2')


    dataset_grey = DSB2018_Dataset()
    dataset_grey.add_nuclei(test_dir, 'test', shuffle = False)
    dataset_grey.prepare()

    dataset_colour = DSB2018_Dataset()
    dataset_colour.add_nuclei(test_dir, 'test', shuffle = False)
    dataset_colour.prepare()

    datasets = [dataset_grey, dataset_colour]
    configs = [config_grey, config_colour]

    predict_nms(configs, datasets) 


def predict_27():
    #submission_DSB2018_768_768_True_12_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180329084920_
    #def train_resnet101_flips_alldata_minimask12_size768():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               image_max_dim = 768,
                               image_min_dim = 768, 
                               images_per_gpu = 1,                              
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_28():
    #
    #NMS combination of def train_resnet101_flips_alldata_minimask12_size768() and train_resnet101_flips_alldata_minimask12()
    
    configs = [mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True}),
               mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               image_max_dim = 768,
                               image_min_dim = 768, 
                               images_per_gpu = 1,                              
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})]

    dataset1 = DSB2018_Dataset()
    dataset1.add_nuclei(test_dir, 'test', shuffle = False)
    dataset1.prepare()

    dataset2 = DSB2018_Dataset()
    dataset2.add_nuclei(test_dir, 'test', shuffle = False)
    dataset2.prepare()

    datasets = [dataset1, dataset2]

    predict_nms(configs, datasets) 


def predict_29():
    #submission_DSB2018_512_512_True_12_28_256_0.5__dim_o-tf-horiz-True-verti-True_0.5_None_20180402101507_
    #train_resnet101_flips_alldata_minimask12_detectionnms0_5():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               detection_nms_threshold = 0.5,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    predict_model(_config, base_test_dataset())


def predict_30():
    #submission_DSB2018_512_512_True_12_28_256_0.3_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180402101810_
    #train_resnet101_flips_alldata_minimask12_double_invert():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset = DSB2018_Dataset(invert_type = 2)
    dataset.add_nuclei(test_dir, 'test')
    dataset.prepare()

    predict_model(_config, dataset)


def predict_31():
    #submission_nms_DSB2018_512_512_True_12_28_256_0.3__dim_o-tf-horiz-True-verti-True_0.5_DSB2018_512_512_True_12_28_256_0.3_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180402105157_
    #NMS combination of def train_resnet101_flips_alldata_minimask12_double_invert() and train_resnet101_flips_alldata_minimask12()
    
    configs = [mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True}),
               mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})]

    dataset1 = DSB2018_Dataset(invert_type = 1)
    dataset1.add_nuclei(test_dir, 'test', shuffle = False)
    dataset1.prepare()

    dataset2 = DSB2018_Dataset(invert_type = 2)
    dataset2.add_nuclei(test_dir, 'test', shuffle = False)
    dataset2.prepare()

    datasets = [dataset1, dataset2]

    predict_nms(configs, datasets) 


def predict_32():
    # submission_DSB2018_512_512_True_12_28_256_0.2_load_image_gt_augment_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180402152908_
    #train_resnet101_flips_alldata_minimask12_double_invert_detectionnms0_2():

    _config = mask_rcnn_config(init_with = 'coco',
                               architecture = 'resnet101',
                               mini_mask_shape = 12,
                               identifier = 'double_invert',
                               detection_nms_threshold = 0.2,
                               augmentation_dict = {'dim_ordering': 'tf',
                                                    'horizontal_flip': True,
                                                    'vertical_flip': True})

    # Training dataset
    dataset = DSB2018_Dataset(invert_type = 2)
    dataset.add_nuclei(test_dir, 'test')
    dataset.prepare()

    predict_model(_config, dataset)


def main():
    predict_32()

if __name__ == '__main__':
    main()
    
    
