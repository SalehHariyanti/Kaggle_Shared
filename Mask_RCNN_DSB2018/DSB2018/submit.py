import os
import datetime
from dsb2018_config import *
from dataset import DSB2018_Dataset
import numpy as np
import model as modellib
import functions as f
from settings import test_dir, submissions_dir
import utils
import dsb2018_utils as du
import scipy
import cv2
from copy import deepcopy
from tqdm import tqdm

import visualize
import matplotlib.pyplot as plt

import train
import getpass
                    
N_SPLITS   = 4
THIS_SPLIT = 1 # from 0 to N_SPLITS-1

def combine_results(_results, N, iou_threshold, voting_threshold, param_dict, use_nms, use_semantic):

    results = []
    for i in range(N):

        img_results = du.concatenate_list_of_dicts([rf[i] for rf in _results])
        if use_nms:
            img_results = reduce_via_nms(img_results, threshold)
        else:
            img_results = reduce_via_voting(img_results, iou_threshold, voting_threshold, param_dict, use_semantic, n_votes = len(_results))

        # Reshape masks
        img_results['masks'] = np.moveaxis(img_results['masks'], 0, -1)
        img_results['class_ids'] = img_results['class_ids'].reshape(-1, )
        img_results['scores'] = img_results['scores'].reshape(-1, )

        results.append(img_results)

    return results


def apply_flips_rotations(_config, images, param_dict):

    images_flip = [images, 
                [np.fliplr(img) for img in images], 
                [np.flipud(img) for img in images],
                [np.flipud(np.fliplr(img)) for img in images],
                [np.rot90(img, 1, (0, 1)) for img in images],
                [np.rot90(img, 3, (0, 1)) for img in images]]

    output = {'images': images_flip,
              'mask_scale': [None] * len(images_flip),
              'fn_reverse': 'reverse_flips_rotations'}

    return output


def reverse_flips_rotations(results_flip, images, use_semantic):

    for i in range(len(images)):

        results_flip[1][i]['masks'] = np.fliplr(results_flip[1][i]['masks'])
        results_flip[2][i]['masks'] = np.flipud(results_flip[2][i]['masks'])
        results_flip[3][i]['masks'] = np.fliplr(np.flipud(results_flip[3][i]['masks']))
        results_flip[4][i]['masks'] = np.rot90(results_flip[4][i]['masks'], -1, (0, 1))
        results_flip[5][i]['masks'] = np.rot90(results_flip[5][i]['masks'], -3, (0, 1))

        if use_semantic:

            results_flip[1][i]['semantic_masks'] = np.fliplr(results_flip[1][i]['semantic_masks'])
            results_flip[2][i]['semantic_masks'] = np.flipud(results_flip[2][i]['semantic_masks'])
            results_flip[3][i]['semantic_masks'] = np.fliplr(np.flipud(results_flip[3][i]['semantic_masks']))
            results_flip[4][i]['semantic_masks'] = np.rot90(results_flip[4][i]['semantic_masks'], -1, (0, 1))
            results_flip[5][i]['semantic_masks'] = np.rot90(results_flip[5][i]['semantic_masks'], -3, (0, 1))

        # Recalculate bboxes
        for j in range(len(results_flip)):
            results_flip[j][i]['rois'] = utils.extract_bboxes(results_flip[j][i]['masks'])
            # Reshape masks so that they can be concatenated correctly
            results_flip[j][i]['masks'] = np.moveaxis(results_flip[j][i]['masks'], -1, 0)
            if use_semantic:
                results_flip[j][i]['semantic_masks'] = np.moveaxis(results_flip[j][i]['semantic_masks'], -1, 0)

    return results_flip


def apply_scaling(_config, images, param_dict):
    """
    Extract the images that would normally be fed into maskrcnn and rescale from here
    NB: because of how maskrcnn resizes, images are scaled up to reach 
    (_config.IMAGE_MIN_DIM, _config.IMAGE_MAX_DIM) as standard.
    As a result, if we have scales > 1 the images will end up being scaled back down to the 
    maximum anyway. So we can only generate different scaled inputs by scaling downwards.
    Hence default scales set to 0.8, 0.9, 1.
    """

    assert 'scales' in param_dict.keys()
    scales = param_dict['scales']

    # Note: output of utils is: image, window, scale, padding
    model_inputs = [utils.resize_image(img, min_dim = _config.IMAGE_MIN_DIM, max_dim = _config.IMAGE_MAX_DIM, padding = _config.IMAGE_PADDING) for img in images]

    # Take the image window out of the model image inputs (the rest is padding)
    model_images_ex_padding = [x[0][x[1][0] : x[1][2], x[1][1] : x[1][3]] for x in model_inputs]
   
    # Create output
    output = {'images': [model_images_ex_padding for scale in scales],
              'mask_scale': [[scale] * len(images) for scale in scales],
              'fn_reverse': 'reverse_scaling'}
    
    return output


def reverse_scaling(results_scale, images, use_semantic):

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

            if use_semantic:
                results_scale[j][i]['semantic_masks'] = scipy.ndimage.zoom(results_scale[j][i]['semantic_masks'], 
                                                                  (images[i].shape[0] / results_scale[j][i]['semantic_masks'].shape[0], 
                                                                   images[i].shape[1] / results_scale[j][i]['semantic_masks'].shape[1], 
                                                                   1), order = 0)
                # Reshape masks so that they can be concatenated correctly
                results_scale[j][i]['semantic_masks'] = np.moveaxis(results_scale[j][i]['semantic_masks'], -1, 0)


    return results_scale


def maskrcnn_detect_augmentations(_config, model, images, list_fn_apply, threshold, voting_threshold = 0.5, param_dict = {}, use_nms = False, use_semantic = False):
    """
    Augments images subject to list_fn_apply and combines results via 
    non-maximum suppression if use_nms is True, otherwises uses merging + voting
    """

    # Apply the augmentations requested
    # NB: augmentations must include the original (unaugmented) image, if requested
    # fn_apply returns: lists of images and corresponding mask_scale, fn_reverse
    images_info = [globals()[fn_apply](_config, images, param_dict) for fn_apply in list_fn_apply]

    results_augment = []

    for img_info in images_info:
        
        # Each img_info set corresponds to a set of augmentations.
        # It has the following information:
        # 'images': the set of augmented images that we require predictions for (e.g. a set of 6 flips/rotations)
        # 'mask_scale': the mask_scale that we want to apply when resizing images with resize_image_scaled() (if you want to use normal resizing set mask_scale = None)
        # 'fn_reverse': the function to call to reverse the augmentations from the predicted masks

        # Detect
        if use_semantic:
            # In this instance we need to detect with expand_semantic = True, 
            # as because we are combining results over multiple augmentations 
            # it is easier to have one semantic instance for each mask instance
            res = [model.detect(img, verbose=0, mask_scale = mask_scale, expand_semantic = True) for img, mask_scale in zip(img_info['images'], img_info['mask_scale'])]
        else:
            res = [model.detect(img, verbose=0, mask_scale = mask_scale) for img, mask_scale in zip(img_info['images'], img_info['mask_scale'])]

        # Reverse augmentations
        res = globals()[img_info['fn_reverse']](res, images, use_semantic)

        # Append
        for r in res:
            results_augment.extend(res)
    
    # Concatenate lists of results
    results_augment = du.concatenate_list_of_dicts(results_augment) 

    if not isinstance(results_augment, list):
        results_augment = [[results_augment]]

    # Carry out either non-maximum suppression or merge+voting to reduce results_flip for each image to a single set of results
    results = combine_results(results_augment, len(images), threshold, voting_threshold, param_dict, use_nms, use_semantic)

    return results


def maskrcnn_detect(_config, model, images, param_dict = {}, use_semantic = False):

    results = model.detect(images, verbose=0)

    if use_semantic:
        for r in results:
            r['masks'] = combine_semantic(r['rois'], r['scores'], r['masks'], r['semantic_masks'], param_dict)

    return results


def reduce_via_nms(img_results, threshold):
    nms_idx = utils.non_max_suppression(img_results['rois'], img_results['scores'].reshape(-1, ), threshold)
    return du.reduce_dict(img_results, nms_idx)


def reduce_via_voting(img_results, threshold, voting_threshold, param_dict, use_semantic, n_votes):
    """
    Merges masks from different sets of results if their bboxes overlap by > threshold.
    Then takes a pixel-level vote on which pixels should be included in each mask (> voting threshold).
    """
    results = deepcopy(img_results)
   
    # Reduce only if masks exist
    if results['rois'].shape[0] > 0:

        # Combine masks with overlaps greater than threshold
        if use_semantic:
            idx, boxes, scores, masks, n_joins, semantic_masks = du.combine_boxes(results['rois'], results['scores'].reshape(-1, ), np.moveaxis(results['masks'], 0, -1), threshold, np.moveaxis(results['semantic_masks'], 0, -1))
        else:
            idx, boxes, scores, masks, n_joins = du.combine_boxes(results['rois'], results['scores'].reshape(-1, ), np.moveaxis(results['masks'], 0, -1), threshold)

        # Select masks based on voting threshold
        avg_masks = masks / n_votes
        masks = (np.multiply(masks, avg_masks > voting_threshold) > 0).astype(np.int)
        valid_masks = np.sum(masks, axis = (0, 1)) > 0

        # Reduce to masks that are still valid
        idx = idx[valid_masks]
        boxes = boxes[valid_masks]
        masks = masks[:, :, valid_masks]
        scores = scores[valid_masks]

        if use_semantic:
            # Reduce semantic masks according to valid_masks and voting_threshold
            avg_semantic_masks = semantic_masks / n_votes
            semantic_masks = (avg_semantic_masks[:, :, valid_masks] > voting_threshold).astype(np.int)
            # Reduce to single semantic mask
            semantic_masks = (np.sum(semantic_masks, axis = -1) > 0).astype(np.int)

            masks = combine_semantic(boxes, scores, masks, semantic_masks, param_dict)

        #from visualize import plot_multiple_images; plot_multiple_images([np.sum(img_results['masks'], axis = 0), np.sum(masks, axis = 0)], nrows = 1, ncols = 2)
    
        # Assign results to img_results:

        # Reduce dict to the relevant index to capture the relevant fields for anything
        # you haven't changed
        img_results = du.reduce_dict(img_results, idx) if len(idx) > 0 else du.reduce_dict(img_results, 0)

        # Assign newly calculated fields
        img_results['rois'] = boxes
        img_results['masks'] = np.moveaxis(masks, -1, 0)
        img_results['scores'] = scores
        if use_semantic:
            img_results['semantic_masks'] = semantic_masks

    else:
        # No masks predicted

        # Reduce to single semantic mask
        if use_semantic:
            results['semantic_masks'] = (np.sum(results['semantic_masks'], axis = 0) > 0).astype(np.int)

        img_results = results

    return img_results


def combine_semantic(boxes, scores, masks, semantic_masks, param_dict):
    """
    For each semantic_mask pixel that falls within the box of a mask
    we assign that pixel to the mask with the highest score.
    """

    n_dilate = param_dict['n_dilate'] if 'n_dilate' in param_dict else 1
    n_erode = param_dict['n_erode'] if 'n_erode' in param_dict else 0

    if n_dilate > 0 or n_erode > 0:

        # Make a mask of box labels
        box_labels = du.maskrcnn_boxes_to_labels(boxes, scores, semantic_masks.shape)

        # Each mask lies between an eroded version of itself and 
        # a dilated version of itself, the pixels in between 
        # being dictated by the overlap with semantic.
        for i in range(masks.shape[-1]):

            # Step 1: find the overlap with semantic. 
            original_overlap = np.multiply(masks[:, :, i], semantic_masks)

            # Step 2: erode the mask.
            eroded_mask = scipy.ndimage.morphology.binary_erosion(masks[:, :, i], iterations = n_erode) if n_erode > 0 else masks[:, :, i]
            eroded_plus_overlap = ((eroded_mask + original_overlap) > 0).astype(np.int)

            # Step 3: dilate the mask within box boundaries and find overlap with semantic
            dilated_mask = scipy.ndimage.morphology.binary_dilation(masks[:, :, i], iterations = n_dilate) if n_dilate > 0 else masks[:, :, i]
            dilated_overlap = np.multiply(dilated_mask, np.multiply(box_labels == (i + 1), semantic_masks))

            # Step 4: combine: new mask = eroded mask + original overlap + dilated overlap
            masks[:, :, i] = ((eroded_plus_overlap + dilated_overlap) > 0).astype(np.int)

        # from visualize import plot_multiple_images; plot_multiple_images([original_overlap, eroded_mask, eroded_plus_overlap, ((eroded_plus_overlap + dilated_overlap) > 0).astype(np.int), masks[:, :, i]])
        # from visualize import plot_multiple_images; plot_multiple_images([original_overlap, eroded_mask, dilated_mask, ((eroded_plus_overlap + dilated_overlap) > 0).astype(np.int), masks[:, :, i], np.abs(((eroded_plus_overlap + dilated_overlap) > 0).astype(np.int) - masks[:,:,i])], nrows = 2, ncols = 3)

    return masks


def create_model(_config, model_name, epoch = None):
    # Recreate the model in inference mode
    model = getattr(modellib, model_name)(mode="inference", 
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


def predict_model(_config, dataset, model_name='MaskRCNN', epoch = None, 
                  augment_flips = False, augment_scale = False, 
                  param_dict = {},
                  nms_threshold = 0.3, voting_threshold = 0.5,
                  use_semantic = False,
                  img_pad = 0, dilate = False, 
                  save_predictions = False, create_submission = True):

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, _config.NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

    # Recreate the model in inference mode
    model = create_model(_config, model_name, epoch)
      
    ImageId = []
    EncodedPixels = []
    
    list_fn_apply = [] + (['apply_flips_rotations'] if augment_flips else []) + (['apply_scaling'] if augment_scale else [])
    
    if dilate:
        n_dilate = param_dict['n_dilate'] if 'n_dilate' in param_dict else 1

    # NB: we need to predict in batches of _config.BATCH_SIZE
    # as there are layers within the model that have strides dependent on this.
    for i in tqdm(range(0, len(dataset.image_ids), _config.BATCH_SIZE)):
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
        if len(list_fn_apply) > 0:
            r = maskrcnn_detect_augmentations(_config, model, images, list_fn_apply, 
                                              threshold = nms_threshold, voting_threshold = voting_threshold, 
                                              param_dict = param_dict, 
                                              use_nms = False, use_semantic = use_semantic)
        else:
            r = maskrcnn_detect(_config, model, images, param_dict = param_dict, use_semantic = use_semantic) 

        # Reduce to N images
        for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

            if j < N:   

                masks = r[j]['masks'] #[H, W, N] instance binary masks
                scores = r[j]['scores']
                boxes = r[j]['rois']

                if img_pad > 0:

                    if use_semantic:
                        r[j]['semantic_masks'] = r[j]['semantic_masks'][img_pad : -img_pad, img_pad : -img_pad]

                    masks = masks[img_pad : -img_pad, img_pad : -img_pad]
                    valid = np.sum(masks, axis = (0, 1)) > 0
                    masks = masks[:, :, valid]

                    r[j]['masks'] = masks
                    r[j]['scores'] = r[j]['scores'][valid]
                    r[j]['class_ids'] = r[j]['class_ids'][valid]
                    r[j]['rois'] = r[j]['rois'][valid]
   
                if dilate:

                    # Dilate masks within boundary box perimeters
                    box_labels = du.maskrcnn_boxes_to_labels(boxes, scores, masks.shape[:2])
                    dilated_masks = []
                    for i in range(masks.shape[-1]):
                        dilated_mask = scipy.ndimage.morphology.binary_dilation(masks[:, :, i], iterations = n_dilate)
                        #from visualize import plot_multiple_images, image_with_masks;                   
                        #plot_multiple_images([image_with_masks(masks[:, :, i], [box_labels == (i + 1)]), np.multiply(box_labels == (i + 1), dilated_mask)])
                        dilated_masks.append(np.multiply(box_labels == (i + 1), dilated_mask))

                    masks = np.stack(dilated_masks, axis = -1)

                img_name = dataset.image_info[idx]['name']
        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, scores)
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if False:
                    class_names = ['background', 'nucleus']
                    visualize.display_instances((images[j] * 255).astype(np.uint8), r[j]['rois'], r[j]['masks'], r[j]['class_ids'], class_names, r[j]['scores'], figsize = (8, 8))
                    
                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:
        submission_filename = os.path.join(submissions_dir, '_'.join(('submission', _config.NAME, str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv')))
        f.write2csv(submission_filename, ImageId, EncodedPixels)
        return submission_filename

    return ImageId, EncodedPixels


def predict_multiple_concat(configs, datasets, model_names, epoch = None, 
                  augment_flips = False, augment_scale = False, 
                  param_dict = {},
                  use_semantic = False,
                  nms_threshold = 0.3, voting_threshold = 0.5,
                  img_pad = 0, dilate = False, 
                  save_predictions = False, create_submission = True):

    ImageId = []
    EncodedPixels = []

    for _config, model_name, dataset in zip(configs, model_names, datasets):
        
        _ImageId, _EncodedPixels = predict_model(_config, dataset, model_name = model_name, epoch = epoch, 
                                                  augment_flips = augment_flips, augment_scale = augment_scale, 
                                                  param_dict = param_dict,
                                                  nms_threshold = nms_threshold, voting_threshold = voting_threshold,
                                                  img_pad = img_pad, dilate = dilate, 
                                                  save_predictions = save_predictions, create_submission = False)
        ImageId += _ImageId
        EncodedPixels += _EncodedPixels

    if create_submission:   

        submission_filename = os.path.join(
            submissions_dir, '_'.join(('submission', 
            configs[0].NAME,
            str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv')))

        f.write2csv(submission_filename, ImageId, EncodedPixels)
        return submission_filename

    return ImageId, EncodedPixels


def predict_voting(configs, datasets, model_names, epochs = None, 
                  augment_flips = False, augment_scale = False, 
                  param_dict = {},
                  use_semantic = False,
                  nms_threshold = 0.3, voting_threshold = 0.5,
                  img_pad = 0, dilate = False, 
                  save_predictions = False, create_submission = True):
    """
    Predicts an ensemble over multiple models via voting
    Presently assumes that model_name/augment_flips/scale/param_dict/threshold/use_semantic are the same 
    for all models you want to ensemble. Need to reformat to make these specific to each model.
    Allows for cases where a single model is made up of multiple submodels that apply to different images.
    """

    # Generalise the format of configs and datasets to cater for cases where a single model set may be
    # made up of multiple models/datasets
    configs = [_config if isinstance(_config, list) else [_config] for _config in configs]
    datasets = [dataset if isinstance(dataset, list) else [dataset] for dataset in datasets]
    model_names = [model_name if isinstance(model_name, list) else [model_name] for model_name in model_names]
    epochs = [epoch if isinstance(epoch, list) else [epoch] for epoch in epochs] if epochs is not None else [[None for d in dataset] for dataset in datasets]
    config_batch_sizes = [[c.BATCH_SIZE for c in _config] for _config in configs]
    batch_size = max([max([b for b in _config_batch_size]) for _config_batch_size in config_batch_sizes])

    # Create the models
    models = [[create_model(c, m, e) for c, e, m in zip(_config, epoch, model_name)] for _config, epoch, model_name in zip(configs, epochs, model_names)]

    # Create a mapping for each model of image_path: model index
    model_infos = merge_model_info(datasets)

    # Make sure that you have a full set of model mappings for each model set
    assert np.all([len(m) == len(model_infos[0]) for m in model_infos[1:]])

    img_paths = np.array(list(model_infos[0].keys()))
    n_images = len(img_paths)

    # Set up holders for the submission rles which you will accumulate
    ImageId = []
    EncodedPixels = []

    list_fn_apply = [] + (['apply_flips_rotations'] if augment_flips else []) + (['apply_scaling'] if augment_scale else [])
   
    # NB: we need to predict in batches of _config.BATCH_SIZE
    # as there are layers within the model that have strides dependent on this.
    split_images = list(np.array_split(range(0, n_images, batch_size), N_SPLITS)[THIS_SPLIT])
    print("Running split {} of {}".format(THIS_SPLIT+1,N_SPLITS))
    for i in tqdm(split_images):

        batch_img_paths = img_paths[i : (i + batch_size)]
        if len(batch_img_paths) != batch_size:
            batch_img_paths = np.append(batch_img_paths, batch_img_paths[:(i + batch_size - len(img_paths))])

        images, images_idx = gather_images(datasets, batch_img_paths)

        images_model_set = [[model[_idx] for _idx in idx] for model, idx in zip(models, images_idx)]
        configs_model_set = [[_config[_idx] for _idx in idx] for _config, idx in zip(configs, images_idx)]
        identical_idx = [np.all([id == _idx[0] for id in _idx]) for _idx in images_idx]

        # Run detection
        res = []
        for model, _images, _config, same_model in zip(images_model_set, images, configs_model_set, identical_idx):

            # Check if we can run the whole batch through with one model
            if same_model and _config[0].BATCH_SIZE == batch_size:

                # Run detection
                if len(list_fn_apply) > 0:
                    r = maskrcnn_detect_augmentations(_config[0], model[0], _images, list_fn_apply, 
                                                      threshold = nms_threshold, voting_threshold = voting_threshold, 
                                                      param_dict = param_dict, 
                                                      use_nms = False, use_semantic = use_semantic)
                else:
                    r = maskrcnn_detect(_config[0], model[0], _images, param_dict = param_dict, use_semantic = use_semantic) 

            else:

                # The batch needs to be split into individual models
                r = []
                for _model, c, img in zip(model, _config, _images):

                    # Artifically expand the batch if required by batch_size
                    batch_img = [img] if c.BATCH_SIZE == 1 else [img] * c.BATCH_SIZE

                    # Run detection
                    if len(list_fn_apply) > 0:
                        prediction = maskrcnn_detect_augmentations(c, _model, batch_img, list_fn_apply, 
                                                            threshold = nms_threshold, voting_threshold = voting_threshold, 
                                                            param_dict = param_dict, 
                                                            use_nms = False, use_semantic = use_semantic)
                    else:
                        prediction = maskrcnn_detect(c, _model, batch_img, param_dict = param_dict, use_semantic = use_semantic)

                    prediction = prediction[0] 

                    r.append(prediction)

            # r now contains the results for the images in the batch
            res.append(r)
        
        # Reduce to N images
        for j, idx in enumerate(range(i, i + batch_size)):      

            if idx < n_images:   

                # Get masks via voting
                
                # First reshape masks so that they can be concatenated:
                for r in res:
                    r[j]['masks'] = np.moveaxis(r[j]['masks'], -1, 0)
                    if use_semantic:
                        # semantic_masks is flat. We need to expand to the r[j]['masks'] dimensions
                        r[j]['semantic_masks'] = np.stack([r[j]['semantic_masks']] * max(1, r[j]['masks'].shape[0]), axis = 0)
                
                # Concatenate
                img_results = du.concatenate_list_of_dicts([r[j] for r in res])

                # Reduce via voting
                img_results = reduce_via_voting(img_results, nms_threshold, voting_threshold, param_dict, use_semantic = use_semantic, n_votes = len(models))

                # Reshape 
                img_results['masks'] = np.moveaxis(img_results['masks'], 0, -1)
                img_results['class_ids'] = img_results['class_ids'].reshape(-1, )
                img_results['scores'] = img_results['scores'].reshape(-1, )

                img_name = os.path.splitext(os.path.split(batch_img_paths[j])[-1])[0]
        
                # Create submission rle entry
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(img_results['masks'], img_name, img_results['scores'])
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

    if create_submission:
        submission_filename = os.path.join(
            submissions_dir, 
            '_'.join(
                ('submission_ensemble', datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '{}of{}'.format(THIS_SPLIT+1,N_SPLITS), '.csv')))

        f.write2csv(submission_filename, ImageId, EncodedPixels)


def merge_model_info(datasets):
    """
    Create a single dictionary for each model set that 
    holds the model index that each image needs to refer to
    """
    model_info = [{}] * len(datasets)
    for d, dataset in enumerate(datasets):
        # We have multiple models/datasets making up the full set
        for j in range(len(dataset)):
            for i in range(len(dataset[j].image_ids)):
                model_info[d][dataset[j].image_info[i]['path']] = j 

    return model_info


def index_by_path(datasets, img_path):

    img_path_idx = [None] * len(datasets)

    for i, dataset in enumerate(datasets):
        for j, d in enumerate(dataset):
            d_paths = [image_info['path'] for image_info in d.image_info]
            if img_path in d_paths:
                img_path_idx[i] = (j, d_paths.index(img_path))
                continue
    return img_path_idx


def gather_images(datasets, batch_img_paths):
    """
    For a given batch of image paths, get the relevant raw data
    from the datasets.
    NB: if 
    """
    n_batch = len(batch_img_paths)

    images = [[] for d in datasets]
    image_idx = [[] for d in datasets]

    for img_path in batch_img_paths:

        img_path_idx = index_by_path(datasets, img_path) 

        for j, path_idx in enumerate(img_path_idx):

            images[j].extend(load_dataset_images(datasets[j][path_idx[0]], path_idx[1], 1))
            image_idx[j].append(path_idx[0]) # the model/dataset that the image is mapped to

    return images, image_idx


def save_model_predictions(save_dir, EncodedPixels_batch, mask_shape, image_info):
    """
    Saves labels from predictions
    """
    if EncodedPixels_batch != ['']:
        labels, masks = du.labels_from_rles(EncodedPixels_batch, mask_shape)
    else:
        labels = np.zeros(mask_shape)
                    
    mosaic_id = image_info['mosaic_id'] if 'mosaic_id' in image_info else 'None'
    mosaic_position = image_info['mosaic_position'] if 'mosaic_position' in image_info else 'None'
    save_filename = os.path.join(save_dir, '_'.join((str(mosaic_id), str(mosaic_position), image_info['name'], '.npy')))

    np.save(save_filename, labels)

    return 


def predict_experiment(fn_experiment, fn_predict = 'predict_model', **kwargs):

    if isinstance(fn_experiment, list):
        _config, dataset, model_name = [], [], []
        for fn in fn_experiment:
            expt_output = fn(training=False)
            _config.append(expt_output[0])
            dataset.append(expt_output[1])
            model_name.append(expt_output[2])
    else:
        _config, dataset, model_name = fn_experiment(training=False)
    submission_filename = globals()[fn_predict](_config, dataset, model_name, **kwargs)

    if submission_filename is not None:
        epoch = 'last' if 'epoch' not in kwargs else kwargs['epoch']
        print("\nkaggle competitions submit -f {} -m 'experiment {} @ epoch {}'".format(
            submission_filename,
            fn_experiment.__name__,
            epoch
            )
        )


def main():

        predict_experiment([train.train_resnet101_semantic,
                            train.train_resnet50_semantic,
                            train.train_resnet101_semantic_maskcount_balanced,
                            train.train_resnet50_semantic_maskcount_balanced,
                            #train.train_resnet101_semantic_maskcount_balanced_gan,
                            #train.train_resnet50_semantic_maskcount_balanced_gan,
                            #train.train_resnet50_semantic_gan
                            ],
                           'predict_voting',
                            augment_flips = True, augment_scale = True,
                            nms_threshold = 0.5, voting_threshold = 0.5,
                            param_dict = {'scales': [0.85, 0.9, 0.95],
                                            'n_dilate': 1,
                                            'n_erode': 0},
                            use_semantic = True)


        predict_experiment([train.train_resnet101_semantic_b_w_colour,
                            train.train_resnet50_semantic_b_w_colour,
                            train.train_resnet101_semantic_b_w_colour_maskcount_balanced,
                            train.train_resnet50_semantic_b_w_colour_maskcount_balanced,
                            #train.train_resnet101_semantic_b_w_colour_maskcount_balanced_gan,
                            #train.train_resnet50_semantic_b_w_colour_maskcount_balanced_gan,
                            #train.train_resnet50_semantic_b_w_colour_gan
                            ],
                           'predict_voting',
                            augment_flips = True, augment_scale = True,
                            nms_threshold = 0.5, voting_threshold = 0.5,
                            param_dict = {'scales': [0.85, 0.9, 0.95],
                                            'n_dilate': 1,
                                            'n_erode': 0},
                            use_semantic = True)



if __name__ == '__main__':
    main()
    
    
