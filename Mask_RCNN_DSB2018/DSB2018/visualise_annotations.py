import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv
from visualize import *
import dsb2018_utils as du
import imageio

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def visualise_annotations(source_dirs, n = 2, target_colour = None):

    # Extract filenames for images/masks
    dirs = [os.path.join(dir, f) for dir in source_dirs for f in os.listdir(dir)]
    files = [os.path.join(f, 'images', ''.join((os.path.split(f)[-1], '.png'))) for f in dirs]
    masks = [[os.path.join(f, 'masks', m) for m in os.listdir(os.path.join(f, 'masks'))] if os.path.exists(os.path.join(f, 'masks')) else None for f in dirs]

    # Reduce to a target colour if requested
    if target_colour is not None:
        from dataset import get_ids
        colour_id, _ = get_ids(files)
        valid_idx = np.argwhere(du.ismember(colour_id, np.array(target_colour))).reshape(-1,)
        files = [files[idx] for idx in valid_idx]
        masks = [masks[idx] for idx in valid_idx]

    img_list = []
    counter = 0
    for f, m in zip(files, masks):

        img = load_img(f)
        masks = np.stack([imageio.imread(path) for path in m], axis = -1)
        labels = du.maskrcnn_mask_to_labels(masks)
        counter += 1

        if counter > n:
            # Display
            plot_multiple_images(img_list, nrows = n, ncols = 2)
            # Reset
            counter = 0
            img_list = []
        else:
            img_list.extend([img, image_with_labels(img, labels)])

    return


def visualise_mosaic_annotations(source_dirs, n = 2):

    # Extract filenames for images/masks
    files = [os.path.join(_dir, f) for _dir in source_dirs for f in os.listdir(_dir) if os.path.splitext(f)[-1] != '.npz']
    masks = [os.path.join(_dir, f) for _dir in source_dirs for f in os.listdir(_dir) if os.path.splitext(f)[-1] == '.npz']

    img_list = []
    counter = 0
    for f, m in zip(files, masks):

        img = load_img(f)
        masks = np.load(m)
        labels = du.maskrcnn_mask_to_labels(masks)
        counter += 1

        if counter > n:
            # Display
            plot_multiple_images(img_list, nrows = n, ncols = 2)
            # Reset
            counter = 0
            img_list = []
        else:
            img_list.extend([img, image_with_labels(img, labels)])

    return


def main():
    #visualise_annotations([os.path.join(base_dir, 'train')])
    visualise_mosaic_annotations([os.path.join(base_dir, 'train_mosaics')])
    
if __name__ == '__main__':
    main()