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

import train

                    
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

    # Carry out non-maximum suppression to reduce results_scale for each image to a single set of results
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


def predict_tiled_model(_config, dataset, epoch = None, tile_threshold = 0, grid_shape = (2, 2), nms_threshold = 0.3, nms_tiles = False):
    """
    Predict masks for each image by splitting the original image up into tiles,
    making predictions for these, and subsequently stitching them back together.
    Only predict via tiling if the image area is > tile_threshold. Otherwise detect as normal.
    """

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


    f.write2csv(os.path.join(submissions_dir, '_'.join(('submission', 
                                                        _config.NAME, 
                                                        str(epoch), 
                                                        str(grid_shape[0]), 
                                                        str(nms_threshold), 
                                                        datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_scaled_model(_config, dataset, epoch = None, nms_threshold = 0.3):
    """
    Predict in two phases:
    1) predict as normal and estimate avg nucleus size
    2) rescale test images according to the model's training avg nucleus size and predict again
    """
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

    f.write2csv(os.path.join(submissions_dir, '_'.join(('submission_scaled', _config.NAME, str(epoch), datetime.datetime.now().strftime('%Y%m%d%H%M%S'), '.csv'))), ImageId, EncodedPixels)


def predict_experiment(fn_experiment, fn_predict = 'predict_model'):
    _config, dataset = fn_experiment(training=False)
    globals()[fn_predict](_config, dataset)


def main():
    predict_experiment(train.train_resnet101_flips_all_rots_data_minimask12_detectionnms0_3_mosaics, 'predict_model')


if __name__ == '__main__':
    main()
    
    
