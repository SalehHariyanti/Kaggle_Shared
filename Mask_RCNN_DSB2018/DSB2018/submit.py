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
from copy import deepcopy

import visualize
import matplotlib.pyplot as plt

import train
import getpass
                    
def base_test_dataset():
    dataset = DSB2018_Dataset()
    dataset.add_nuclei(test_dir, 'test')
    dataset.prepare()
    return dataset


def combine_results(_results, N, iou_threshold, voting_threshold, use_nms):

    results = []
    for i in range(N):

        img_results = du.concatenate_list_of_dicts([rf[i] for rf in _results])
        if use_nms:
            img_results = reduce_via_nms(img_results, threshold)
        else:
            img_results = reduce_via_voting(img_results, iou_threshold, voting_threshold, n_votes = len(_results))

        # Reshape masks
        img_results['masks'] = np.moveaxis(img_results['masks'], 0, -1)
        img_results['class_ids'] = img_results['class_ids'].reshape(-1, )
        img_results['scores'] = img_results['scores'].reshape(-1, )

        results.append(img_results)

    return results


def apply_flips_rotations(_config, images):

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


def reverse_flips_rotations(results_flip, images):

    for i in range(len(images)):
        results_flip[1][i]['masks'] = np.fliplr(results_flip[1][i]['masks'])
        results_flip[2][i]['masks'] = np.flipud(results_flip[2][i]['masks'])
        results_flip[3][i]['masks'] = np.fliplr(np.flipud(results_flip[3][i]['masks']))
        results_flip[4][i]['masks'] = np.rot90(results_flip[4][i]['masks'], -1, (0, 1))
        results_flip[5][i]['masks'] = np.rot90(results_flip[5][i]['masks'], -3, (0, 1))

        # Recalculate bboxes
        for j in range(len(results_flip)):
            results_flip[j][i]['rois'] = utils.extract_bboxes(results_flip[j][i]['masks'])
            # Reshape masks so that they can be concatenated correctly
            results_flip[j][i]['masks'] = np.moveaxis(results_flip[j][i]['masks'], -1, 0)

    return results_flip


def apply_scaling(_config, images, scales = [0.8, 0.9, 1]):
    """
    Extract the images that would normally be fed into maskrcnn and rescale from here
    NB: because of how maskrcnn resizes, images are scaled up to reach 
    (_config.IMAGE_MIN_DIM, _config.IMAGE_MAX_DIM) as standard.
    As a result, if we have scales > 1 the images will end up being scaled back down to the 
    maximum anyway. So we can only generate different scaled inputs by scaling downwards.
    Hence default scales set to 0.8, 0.9, 1.
    """

    # Note: output of utils is: image, window, scale, padding
    model_inputs = [utils.resize_image(img, min_dim = _config.IMAGE_MIN_DIM, max_dim = _config.IMAGE_MAX_DIM, padding = _config.IMAGE_PADDING) for img in images]
    # Take the image window out of the model image inputs (the rest is padding)
    model_images_ex_padding = [x[0][x[1][0] : x[1][2], x[1][1] : x[1][3]] for x in model_inputs]
   
    # Create output
    output = {'images': [model_images_ex_padding for scale in scales],
              'mask_scale': [[scale] * len(images) for scale in scales],
              'fn_reverse': 'reverse_scaling'}
    
    return output


def reverse_scaling(results_scale, images):

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


    return results_scale


def maskrcnn_detect_augmentations(_config, model, images, list_fn_apply, threshold, voting_threshold = 0.5, use_nms = False):
    """
    Augments images subject to list_fn_apply and combines results via 
    non-maximum suppression if use_nms is True, otherwises uses merging + voting
    """

    # Apply the augmentations requested
    # NB: augmentations must include the original (unaugmented) image, if requested
    # fn_apply returns: lists of images and corresponding mask_scale, fn_reverse
    images_info = [globals()[fn_apply](_config, images) for fn_apply in list_fn_apply]

    results_augment = []

    for img_info in images_info:
        
        # Each img_info set corresponds to a set of augmentations.
        # It has the following information:
        # 'images': the set of augmented images that we require predictions for (e.g. a set of 6 flips/rotations)
        # 'mask_scale': the mask_scale that we want to apply when resizing images with resize_image_scaled() (if you want to use normal resizing set mask_scale = None)
        # 'fn_reverse': the function to call to reverse the augmentations from the predicted masks

        # Detect
        res = [model.detect(img, verbose=0, mask_scale = mask_scale) for img, mask_scale in zip(img_info['images'], img_info['mask_scale'])]

        # Reverse augmentations
        res = globals()[img_info['fn_reverse']](res, images)

        # Append
        for r in res:
            results_augment.extend(res)
    
    # Concatenate lists of results
    results_augment = du.concatenate_list_of_dicts(results_augment)

    # Carry out either non-maximum suppression or merge+voting to reduce results_flip for each image to a single set of results
    results = combine_results(results_augment, len(images), threshold, voting_threshold, use_nms)

    return results


def maskrcnn_detect_tiles(model, images, grid_shape = (2, 2), nms_threshold = 0.3, nms_tiles = False):
    """
    Splits images into tiles and combines results.
    If nms_tiles = True, the tiled results are combined with those for the original image.
    If nms_tiles = False, tiled results that are not on the border are used, and combined with results for the original image that do not overlap with tiled masks.
    """
    
    n_images = len(images)
    n_grid = np.product(grid_shape)
    border_pixel_threshold = 5

    # grid_data = (grid, x, y, per_col, per_row, pad_col, pad_row) for each image
    grid_data = [create_tile_template(img.shape, grid_shape) for img in images]

    # Tile original images
    images_tiled = [tile_image(img, grid) for img, grid in zip(images, grid_data)]

    # Reorder so that you have n images x m tiles
    images_tiled = [[images_tiled[j][i] for j in range(n_images)] for i in range(n_grid)]
    
    # Detect (results in orginal format)
    results_tiles = [model.detect(imgs, verbose = 0) for imgs in images_tiled]

    # Combine back from tiles to original
    masks_retiled = [[]] * n_images
    masks_retiled_scores = [[]] * n_images
    masks_retiled_class_ids = [[]] * n_images
    for i in range(n_images):

        for j in range(n_grid):

            # We will be ignoring any masks from the tiles that are too close to borders as these are unreliable.
            # (You will replace these with nuclei from the original mask)
            border_pixels = np.ones(results_tiles[j][i]['masks'].shape)
            border_pixels[border_pixel_threshold : -border_pixel_threshold, border_pixel_threshold : -border_pixel_threshold] = 0

            valid_mask = np.logical_and(np.sum(np.multiply(results_tiles[j][i]['masks'], border_pixels), axis = (0, 1)) == 0,
                                        np.sum(results_tiles[j][i]['masks'], axis = (0, 1)) > 0)

            if False: 
                #Sanity checking...
                visualize.plot_multiple_images([(images[i]*255).astype(np.uint8),
                                                (images[i]*255).astype(np.uint8),
                                                du.maskrcnn_mask_to_labels(results_tiles[0][i]['masks']),
                                                du.maskrcnn_mask_to_labels(results_tiles[1][i]['masks']),
                                                du.maskrcnn_mask_to_labels(results_tiles[2][i]['masks']),
                                                du.maskrcnn_mask_to_labels(results_tiles[3][i]['masks'])],
                                               nrows = 3, ncols = 2)

                visualize.plot_multiple_images([du.maskrcnn_mask_to_labels(results_tiles[j][i]['masks']), 
                                                du.maskrcnn_mask_to_labels(results_tiles[j][i]['masks'][:, :, valid_mask])], nrows = 1, ncols = 2)

            this_n_masks = np.sum(valid_mask)

            if this_n_masks > 0:

                this_mask = np.zeros(grid_data[i][0].shape + (this_n_masks,), dtype = np.int)
                this_mask[grid_data[i][0] == j + 1] = results_tiles[j][i]['masks'][:, :, valid_mask].reshape(-1, this_n_masks)

                if len(masks_retiled[i]) == 0:
                    masks_retiled[i] = np.moveaxis(this_mask, -1, 0)
                    masks_retiled_scores[i] = results_tiles[j][i]['scores'][valid_mask]
                    masks_retiled_class_ids[i] = results_tiles[j][i]['class_ids'][valid_mask]
                else:
                    masks_retiled[i] = np.concatenate((masks_retiled[i], np.moveaxis(this_mask, -1, 0)))
                    masks_retiled_scores[i] = np.concatenate((masks_retiled_scores[i], results_tiles[j][i]['scores'][valid_mask]))
                    masks_retiled_class_ids[i] = np.concatenate((masks_retiled_class_ids[i], results_tiles[j][i]['class_ids'][valid_mask]))

            else:

                masks_retiled[i] = np.zeros((1, ) + grid_data[i][0].shape, dtype = np.int)
                masks_retiled_scores[i] = np.array([0])
                masks_retiled_class_ids[i] = np.array([0])

    results_retiled = [{'masks': mask_retiled[:, :images[i].shape[0], :images[i].shape[1]], 
                        'scores': mask_retiled_score,
                        'class_ids': mask_retiled_class_id,
                        'rois': utils.extract_bboxes(np.moveaxis(mask_retiled, 0, -1))} 
                        for i, (mask_retiled, mask_retiled_score, mask_retiled_class_id) in enumerate(zip(masks_retiled, masks_retiled_scores, masks_retiled_class_ids))]
        
    # Detect original results
    results_orig = model.detect(images, verbose = 0)
    # Reshape masks so that they are compatible with pu.concatenate_list_of_dicts
    for r in results_orig:
        r['masks'] = np.moveaxis(r['masks'], -1, 0)
        
    # Reduce each image to a single set of results, combining the original with the tiled version
    results = []
    for i in range(len(images)):

        if nms_tiles:
            # Carry out NMS of the original with the tiled version (less borders)
            img_results_retiled = du.concatenate_list_of_dicts([results_orig[i], results_retiled[i]])
            nms_idx = utils.non_max_suppression(img_results_retiled['rois'], img_results_retiled['scores'].reshape(-1, ), nms_threshold)
            img_results_retiled = du.reduce_dict(img_results_retiled, nms_idx)
        else:
            # Use the masks from the original that do not overlap with the tiled version
            # and let the tiled version take preference for non-border nuclei
            overlap = np.sum(np.multiply(results_orig[i]['masks'], np.expand_dims(np.sum(results_retiled[i]['masks'], axis = 0), 0)), axis = (1, 2)) > 0
            valid_orig = overlap == 0

            #visualize.plot_multiple_images([du.maskrcnn_mask_to_labels(np.moveaxis(results_orig[i]['masks'], 0, -1)), du.maskrcnn_mask_to_labels(np.moveaxis(results_orig[i]['masks'][valid_orig, :, :], 0, -1))], nrows = 1, ncols = 2) 

            results_orig[i]['masks'] = results_orig[i]['masks'][valid_orig]
            results_orig[i]['scores'] = results_orig[i]['scores'][valid_orig]
            results_orig[i]['rois'] = results_orig[i]['rois'][valid_orig]
            results_orig[i]['class_ids'] = results_orig[i]['class_ids'][valid_orig]

            img_results_retiled = du.concatenate_list_of_dicts([results_orig[i], results_retiled[i]])

        # Reshape elements as required
        img_results_retiled['masks'] = np.moveaxis(img_results_retiled['masks'], 0, -1)
        img_results_retiled['class_ids'] = img_results_retiled['class_ids'].reshape(-1, )
        img_results_retiled['scores'] = img_results_retiled['scores'].reshape(-1, )

        results.append(img_results_retiled)


    if False:
        # Sanity checking...
        visualize.plot_multiple_images([(images[0] * 255).astype(np.uint8), 
                                    du.maskrcnn_mask_to_labels(np.moveaxis(results_retiled[0]['masks'], 0, -1)),
                                    du.maskrcnn_mask_to_labels(np.moveaxis(results_orig[0]['masks'], 0, -1)),
                                    du.maskrcnn_mask_to_labels(results[0]['masks'])], 
                                    ['img', 'tiled predictions (less borders)', 'orig predictions', 'combined predictions'], 1, 4)

    return results


def reduce_via_nms(img_results, threshold):
    nms_idx = utils.non_max_suppression(img_results['rois'], img_results['scores'].reshape(-1, ), threshold)
    return du.reduce_dict(img_results, nms_idx)


def reduce_via_voting(img_results, threshold, voting_threshold, n_votes):
    """
    Merges masks from different sets of results if their bboxes overlap by > threshold.
    Then takes a pixel-level vote on which pixels should be included in each mask (> voting threshold).
    """
    results = deepcopy(img_results)
   
    # Combine masks with overlaps greater than threshold
    idx, boxes, scores, masks, n_joins = du.combine_boxes(results['rois'], results['scores'].reshape(-1, ), np.moveaxis(results['masks'], 0, -1), threshold)

    # Select masks based on voting threshold
    masks = np.moveaxis(masks, -1, 0)
    avg_masks = masks / n_votes
    masks = (np.multiply(masks, avg_masks > voting_threshold) > 0).astype(np.int)
    valid_masks = np.sum(masks, axis = (1, 2)) > 0

    # Reduce to masks that are still valid
    idx = idx[valid_masks]
    boxes = boxes[valid_masks]
    masks = masks[valid_masks, :, :]
    scores = scores[valid_masks]

    #from visualize import plot_multiple_images; plot_multiple_images([np.sum(img_results['masks'], axis = 0), np.sum(masks, axis = 0)], nrows = 1, ncols = 2)
    
    img_results = du.reduce_dict(img_results, idx)
     
    img_results['rois'] = boxes
    img_results['masks'] = masks
    img_results['scores'] = scores

    return img_results


def tile_image(img, grid):

    grid_template, x, y, per_col, per_row, pad_col, pad_row = grid

    tiled_img = np.zeros(grid_template.shape if img.ndim == 2 else grid_template.shape + (img.shape[-1],))
    tiled_img[:img.shape[0], :img.shape[1]] = img

    if img.ndim == 2:
        tiled_output = [tiled_img[grid_template == i].reshape((x, y)) for i in range(1, int(grid_template.max() + 1))]
    else:
        tiled_output = [np.stack([_tiled_img[grid_template == i].reshape((x, y)) 
                                    for _tiled_img in list(np.moveaxis(tiled_img, -1, 0))], axis = -1) 
                        for i in range(1, int(grid_template.max() + 1))]

    return tiled_output


def create_tile_template(img_shape, grid_shape):

    x, y = int(np.ceil(img_shape[0] / grid_shape[0])), int(np.ceil(img_shape[1] / grid_shape[1]))
    per_col, per_row = grid_shape[0], grid_shape[1]
    pad_col, pad_row = (x * grid_shape[0]) - img_shape[0], (y * grid_shape[1]) - img_shape[1]

    single_tile = np.ones((x, y))

    row = np.concatenate([single_tile * (i + 1) for i in range(per_row)], axis =1)
    grid = np.concatenate([row + i for i in per_row * np.arange(per_col)], axis = 0)

    return grid, x, y, per_col, per_row, pad_col, pad_row


def create_model(_config, model_name, epoch = None):
    # Recreate the model in inference mode
    model = getattr(modellib, model_name)(mode="inference", 
                                config=_config,
                                model_dir=_config.MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")

    print(model)
    dir_preffix = model_name if model_name != 'MaskRCNN' else ''
    if epoch is not None:
        model_path = model.find_last(dir_preffix)[1][:-7] + '00' + str(epoch) + '.h5'
    else:
        model_path = model.find_last(dir_preffix)[1]

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


def combine_mosaics(_orig_names, _orig_mosaic_position, _orig_scores, _orig_labels, _mosaic_labels, _mosaic_scores):

    orig_mosaic_names = [_orig_names[_orig_mosaic_position == 'down_left'][0],
                            _orig_names[_orig_mosaic_position == 'down_right'][0],
                            _orig_names[_orig_mosaic_position == 'up_left'][0],
                            _orig_names[_orig_mosaic_position == 'up_right'][0]]

    orig_mosaic_scores = [_orig_scores[_orig_mosaic_position == 'down_left'][0],
                            _orig_scores[_orig_mosaic_position == 'down_right'][0],
                            _orig_scores[_orig_mosaic_position == 'up_left'][0],
                            _orig_scores[_orig_mosaic_position == 'up_right'][0]]

    orig_mosaic_labels = [_orig_labels[_orig_mosaic_position == 'down_left'][0],
                            _orig_labels[_orig_mosaic_position == 'down_right'][0],
                            _orig_labels[_orig_mosaic_position == 'up_left'][0],
                            _orig_labels[_orig_mosaic_position == 'up_right'][0]]

    # Divide the mosaic prediction into four original predictions
    full_mosaic_labels = [_mosaic_labels[orig_mosaic_labels[1].shape[0]:, :orig_mosaic_labels[0].shape[1]],
                            _mosaic_labels[orig_mosaic_labels[1].shape[0]:, orig_mosaic_labels[0].shape[1]:],
                            _mosaic_labels[:orig_mosaic_labels[1].shape[0], :orig_mosaic_labels[0].shape[1]],
                            _mosaic_labels[:orig_mosaic_labels[1].shape[0]:, orig_mosaic_labels[0].shape[1]:]]

    # Extract the scores relating to each part of the divided mosaic
    full_mosaic_scores = []
    for l in full_mosaic_labels:
        unique_labels = np.unique(l).astype(np.int)
        score_idx = unique_labels[unique_labels > 0] - 1
        full_mosaic_scores.append(_mosaic_scores[score_idx])

    # Combine the two, giving preference to full_mosaic_labels
    output = [combine_labels([fl, l], [fs, s]) for fl, l, fs, s in zip(full_mosaic_labels, 
                                                                        orig_mosaic_labels, 
                                                                        full_mosaic_scores, 
                                                                        orig_mosaic_scores)]
    final_labels = [o[0] for o in output]
    final_scores = [o[1] for o in output]

    return final_labels, final_scores, orig_mosaic_names


def combine_labels(label_list, score_list, border_threshold = 5):
    """
    label_list, score_list: length 2
    First element corresponds to your baseline predictions.
    The second corresponds to the predictions you wish to use provided
    they do not overlap with your baseline predictions and that they are not within
    border_threshold of the edge.
    """

    final_labels = label_list[0].copy()

    overlap = np.multiply(label_list[0] > 0, label_list[1] > 0)

    border = np.ones_like(overlap); border[border_threshold : -border_threshold, border_threshold : -border_threshold] = 0

    # A label from the second set is invalid if it overlaps with the first set, or if it broaches the border
    invalid_overlap = np.unique(label_list[1][np.multiply(label_list[1], overlap) > 0])
    invalid_border = np.unique(label_list[1][np.multiply(label_list[1], border) > 0])

    # Zero out invalid labels
    label_f = label_list[1].flatten()
    label_f[du.ismember(label_f, np.concatenate((invalid_overlap, invalid_border)), index_requested = False)] = 0
    # Visual:
    if False:
        from visualize import plot_multiple_images; plot_multiple_images(label_list + [label_f.reshape(final_labels.shape)], nrows = 1, ncols = 3)

    # Retrieve the scores corresponding to the labels you have left
    add_labels = np.unique(label_f).astype(np.int)
    add_scores = score_list[1][add_labels[add_labels > 0] - 1]

    # Artifically add final_labels.max() to the additional labels so that they don't clash when combined
    label_f[label_f > 0] = label_f[label_f > 0] + final_labels.max()

    # Create final labels
    final_labels = final_labels + label_f.reshape(final_labels.shape)
    # Rebase to start from 1 -> N without gaps
    final_labels = du.create_id(final_labels.flatten()).reshape(final_labels.shape)
    # Create final scores
    final_scores = np.concatenate((score_list[0], add_scores))

    return final_labels, final_scores


def predict_model(_config, dataset, model_name='MaskRCNN', epoch = None, 
                  augment_flips = False, augment_scale = False, 
                  nms_threshold = 0.3, voting_threshold = 0.5,
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
        list_fn_apply = [] + (['apply_flips_rotations'] if augment_flips else []) + (['apply_scaling'] if augment_scale else [])
        if len(list_fn_apply) > 0:
            r = maskrcnn_detect_augmentations(_config, model, images, list_fn_apply, threshold = nms_threshold, voting_threshold = voting_threshold, use_nms = False)
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
                    
                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:
        submission_filename = os.path.join(submissions_dir, '_'.join(('submission', _config.NAME, str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv')))
        f.write2csv(submission_filename, ImageId, EncodedPixels)
        return submission_filename


def predict_multiple(configs, datasets, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3, save_predictions = False, create_submission = True):

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, configs[0].NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

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

                    if save_predictions:
                        # Extract final masks from EncodedPixels_batch here and save
                        # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                        save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:                   

        f.write2csv(os.path.join(submissions_dir, '_'.join(('submission', 
                                                        '_'.join([_config.NAME for _config in configs]), 
                                                        str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_nms(configs, datasets, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3, save_predictions = False, create_submission = True):

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, configs[0].NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

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
                    
                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:
        f.write2csv(os.path.join(submissions_dir, '_'.join(('submission_nms', 
                                                    '_'.join([_config.NAME for _config in configs]), 
                                                    str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_labels(_config, dataset, epoch = None):

    # Recreate the model in inference mode
    model = create_model(_config, epoch)
    
    labels, scores = [], []
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
        r = model.detect(images, verbose=0)
        
        # Reduce to N images
        for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

            if j < N:   

                masks = r[j]['masks'] #[H, W, N] instance binary masks
                
                labels.append(du.maskrcnn_mask_to_labels(masks))
                scores.append(r[j]['scores'])
                
    return np.array(labels), np.array(scores)


def predict_mosaics_plus_originals(configs, datasets, epoch = None, augment_flips = False, augment_scale = False, nms_threshold = 0.3, save_predictions = False, create_submission = True):
    """
    Predict mosaics (config/dataset 0); deconstruct to originals.
    Predict originals (config/dataset 1).
    Combine mosaics with originals that have no overlap with mosaics (excluding originals that cross an edge... edge cases unreliable).
    """

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, configs[0].NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

    assert len(configs) == len(datasets) == 2

    ImageId = []
    EncodedPixels = []

    # Predict full mosaic images using predict_mosaic_labels + configs/datasets[0]
    mosaic_id = np.array([int(datasets[0].image_info[i]['mosaic_id']) for i in range(len(datasets[0].image_ids))])
    mosaic_labels, mosaic_scores = predict_labels(configs[0], datasets[0], epoch = epoch)

    # Extract mosaic info from the original images
    n_images = len(datasets[1].image_ids)
    orig_names = np.array([datasets[1].image_info[i]['name'] for i in range(n_images)])
    orig_mosaic_id = np.array([datasets[1].image_info[i]['mosaic_id'] for i in range(n_images)])
    orig_mosaic_position = np.array([datasets[1].image_info[i]['mosaic_position'] for i in range(n_images)])

    # Predict original images using predict_labels + configs/datasets[1]
    labels, scores = predict_labels(configs[1], datasets[1], epoch = epoch)

    # Combine mosaic predictions with originals, giving preference to mosaics
    assert np.array_equal(np.unique(mosaic_id), np.unique(orig_mosaic_id))

    for mosaic in np.unique(mosaic_id):

        mosaic_idx = np.argwhere(orig_mosaic_id == mosaic).reshape(-1,)

        if len(mosaic_idx) > 1:
            # Image is part of a mosaic.
            # Order the original predictions according to down-left/down-right/up-left/up-right
            _orig_names = orig_names[mosaic_idx]
            _orig_mosaic_position = orig_mosaic_position[mosaic_idx]
            _orig_labels = labels[mosaic_idx]
            _orig_scores = scores[mosaic_idx]

            final_labels, final_scores, final_names = combine_mosaics(_orig_names, _orig_mosaic_position, _orig_scores, _orig_labels, 
                                                                      mosaic_labels[mosaic_id == mosaic][0], mosaic_scores[mosaic_id == mosaic][0])

            # Add to predictions
            for fl, fs, img_name, pos in zip(final_labels, final_scores, final_names, ['down_left', 'down_right', 'up_left', 'up_right']):

                masks = du.maskrcnn_labels_to_mask(fl)        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, fs)
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    image_info = {'mosaic_id': mosaic, 'mosaic_position': pos, 'name': img_name}
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], image_info)

        else:
            # Predictions are the same as image not part of a mosaic
            masks = du.maskrcnn_labels_to_mask(labels[mosaic_idx][0])
            img_name = datasets[1].image_info[mosaic_idx[0]]['name']
        
            ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, scores[mosaic_idx][0])
            ImageId += ImageId_batch
            EncodedPixels += EncodedPixels_batch

            if save_predictions:
                # Extract final masks from EncodedPixels_batch here and save
                # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], datasets[1].image_info[mosaic_idx[0]])

    if create_submission:
        f.write2csv(os.path.join(submissions_dir, '_'.join(('submission_mosaics', 
                                                    configs[0].NAME, 
                                                    str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_tiled_model(_config, dataset, epoch = None, tile_threshold = 0, grid_shape = (2, 2), nms_threshold = 0.3, nms_tiles = False, save_predictions = False, create_submission = True):
    """
    Predict masks for each image by splitting the original image up into tiles,
    making predictions for these, and subsequently stitching them back together.
    Only predict via tiling if the image area is > tile_threshold. Otherwise detect as normal.
    """

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, _config.NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

    # Recreate the model in inference mode
    model = create_model(_config, epoch)
      
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
                images.append(dataset.load_image(dataset.image_ids[idx]))
            else:
                images.append(images[-1])

        # Run detection
        # maskrcnn_detect_tiles if img_size > threshold else regular detection
        # NB: We detect for all images in each case as we need to keep batch sizes consistent
            
        img_size = [np.product(img.shape[:2]) for img in images]
        detect_tiles = [s > tile_threshold for s in img_size]

        if np.any(detect_tiles):
            r_tiles = maskrcnn_detect_tiles(model, images, grid_shape = grid_shape, nms_threshold = nms_threshold, nms_tiles = nms_tiles)

        if not np.all(detect_tiles):
            r_orig = model.detect(images)

        r = [r_tiles[j] if detect_tiles[j] else r_orig[j] for j in range(len(detect_tiles))]

        # Reduce to N images
        for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

            if j < N:   

                masks = r[j]['masks'] #[H, W, N] instance binary masks
                img_name = dataset.image_info[idx]['name']
        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, r[j]['scores'])
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:
        f.write2csv(os.path.join(submissions_dir, '_'.join(('submission', 
                                                        _config.NAME, 
                                                        str(epoch), 
                                                        str(grid_shape[0]), 
                                                        str(nms_threshold), 
                                                        datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_scaled_model(_config, dataset, epoch = None, nms_threshold = 0.3, save_predictions = False, create_submission = True):
    """
    Predict in two phases:
    1) predict as normal and estimate avg nucleus size
    2) rescale test images according to the model's training avg nucleus size and predict again
    """

    # Create save_dir
    if save_predictions:
        save_dir = os.path.join(data_dir, _config.NAME, '_'.join(('submission', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))))
        os.makedirs(save_dir)

    # Recreate the model in inference mode
    model = create_model(_config, epoch)

    # Retrieve the avg mask size
    assert _config.mask_size_filename is not None
    model_mask_size = np.load(_config.mask_size_filename)
    model_mask_size = np.mean(model_mask_size[np.logical_not(np.isnan(model_mask_size))])
      
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
                images.append(dataset.load_image(dataset.image_ids[idx]))
            else:
                images.append(images[-1])

        # Run detection
        _r = model.detect(images, verbose=0)
        
        # Estimate the mask size 
        avg_mask_size = []
        for j in range(_config.BATCH_SIZE):                 
            masks = _r[j]['masks'] 
            #NB: need to resize to model input scale so that is comparable with recorded model_mask_size
            _, window, scale, padding = utils.resize_image(images[j], min_dim = _config.IMAGE_MIN_DIM, max_dim = _config.IMAGE_MAX_DIM, padding = _config.IMAGE_PADDING)
            resized_masks = utils.resize_mask(masks, scale, padding)
            avg_mask_size.append(np.mean(np.sum(resized_masks, axis = (0, 1))))

        # Detect again, but with images scaled according to predicted mask size
        r = model.detect(images, verbose = 0, mask_scale = [np.sqrt(model_mask_size / x) for x in avg_mask_size])

        for j, idx in enumerate(range(i, i + _config.BATCH_SIZE)):      

            if j < N:   

                masks = r[j]['masks'] #[H, W, N] instance binary masks
                img_name = dataset.image_info[idx]['name']
        
                ImageId_batch, EncodedPixels_batch = f.numpy2encoding_no_overlap_threshold(masks, img_name, r[j]['scores'])
                ImageId += ImageId_batch
                EncodedPixels += EncodedPixels_batch

                if save_predictions:
                    # Extract final masks from EncodedPixels_batch here and save
                    # using filename: (mosaic_id)_(mosaic_position)_(img_name).npy
                    save_model_predictions(save_dir, EncodedPixels_batch, masks.shape[:2], dataset.image_info[idx])


    if create_submission:
        f.write2csv(os.path.join(submissions_dir, '_'.join(('submission_scaled', _config.NAME, str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


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
    if getpass.getuser() == 'antor':
        predict_experiment(train.train_resnet101_flipsrot_minimask12_double_invert_semantic, 'predict_model')
    else:
        #predict_experiment(train.train_resnet101_flips_all_rots_data_minimask12_mosaics_nsbval, 'predict_model', create_submission = False, save_predictions = True)
        predict_experiment(train.train_resnet101_flips_alldata_minimask12_double_invert, 'predict_model', 
                           augment_flips = True, augment_scale = True, 
                           nms_threshold = 0.5, voting_threshold = 0.5,
                           create_submission = True, save_predictions = False)

if __name__ == '__main__':
    main()
    
    
