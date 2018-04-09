import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv
import sys
sys.path.append('../')
from visualize import *
from dsb2018_utils import * 

import getpass
USER = getpass.getuser()

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(base_dir, 'test')
submissions_dir = os.path.join(base_dir, 'submissions')
test_mosaics_dir = os.path.join(base_dir, 'test_mosaics')

#############
# Comparing submission outputs



def extract_data(file):

    reader = csv.reader(open(file, 'rt'))

    filename = []
    rle = []
    for i, row in enumerate(reader):
        if i > 0:
            filename.append(row[0])
            rle.append(row[1])

    filename = np.array(filename)
    u_filename = np.unique(filename)

    files_out = np.empty((len(u_filename), ), dtype = object)
    rles_out = np.empty((len(u_filename), ), dtype = object)
    for i, file in enumerate(u_filename):
        index = np.argwhere(filename == file).reshape(-1,)
        files_out[i] = file
        rles_out[i] = [rle[j] for j in index]

    return files_out, rles_out


def compare_submissions(submission_files):

    submissions_data = [extract_data(file) for file in submission_files]

    submissions_filenames = [x[0] for x in submissions_data]
    submissions_rles = [x[1] for x in submissions_data]

    for i in range(len(submissions_filenames[0])):
        this_file = submissions_filenames[0][i]
        test_img = load_img(os.path.join(test_dir, this_file, 'images', ''.join((this_file, '.png'))), greyscale = True)
        labels = [labels_from_rles(sr[np.argwhere(sf == this_file).reshape(-1,)][0], test_img.shape[:2])[0] for sr, sf in zip(submissions_rles, submissions_filenames)]
        plot_multiple_images([test_img] + [image_with_labels(test_img, l) for l in labels] + [image_with_masks(test_img, labels)], 
                             ['img'] + ['_'.join(('submission', str(i), str(np.max(l)))) for i, l in enumerate(labels)] + ['img_with_masks'], 
                             1, len(labels) + 2)

    return


def mosaics_from_submissions(submission_dir):

    labelfiles = np.array(os.listdir(submission_dir))
    files = np.array([os.path.splitext(l)[0].split('_')[-2] for l in labelfiles])
    mosaic_id = np.array([os.path.splitext(l)[0].split('_')[0] for l in labelfiles])
    mosaic_position = np.array([os.path.splitext(l)[0].replace(f, '').replace(m + '_', '')[:-2] for l, f, m in zip(labelfiles, files, mosaic_id)])

    for mosaic in np.unique(mosaic_id):

        this_index = np.argwhere(mosaic_id == mosaic).reshape(-1, )

        if len(this_index) > 1:

            imgs = [load_img(os.path.join(test_dir, f, 'images', '.'.join((f, 'png')))) for f in files[this_index]]
            labels = [np.load(os.path.join(submission_dir, f)) for f in labelfiles[this_index]]

            shapes = [i.shape[:2] for i in imgs]
            mosaic = np.zeros((shapes[0][0] + shapes[1][0], shapes[0][1] + shapes[1][1]))
            mosaic_img = np.zeros((shapes[0][0] + shapes[1][0], shapes[0][1] + shapes[1][1], imgs[0].shape[-1]))

            for i, idx in enumerate(this_index):
                if mosaic_position[idx] == 'down_left':
                    mosaic[shapes[0][0]:, :shapes[0][1]] = labels[i]
                    mosaic_img[shapes[0][0]:, :shapes[0][1]] = imgs[i]
                elif mosaic_position[idx] == 'down_right':
                    mosaic[shapes[0][0]:, shapes[0][1]:] = labels[i]
                    mosaic_img[shapes[0][0]:, shapes[0][1]:] = imgs[i]
                elif mosaic_position[idx] == 'up_left':
                    mosaic[:shapes[0][0], :shapes[0][1]] = labels[i]
                    mosaic_img[:shapes[0][0], :shapes[0][1]] = imgs[i]
                else:
                    mosaic[:shapes[0][0], shapes[0][1]:] = labels[i]
                    mosaic_img[:shapes[0][0], shapes[0][1]:] = imgs[i]

            plot_multiple_images([mosaic_img, mosaic, image_with_labels(mosaic_img, mosaic)],
                                 ['image', 'labels', 'img with labels'],
                                 1, 3)


def masks_for_test_mosaics(submission_dir):

    labelfiles = np.array(os.listdir(submission_dir))
    files = np.array([os.path.splitext(l)[0].split('_')[-2] for l in labelfiles])
    imgfiles = [os.path.join(test_mosaics_dir, f) for f in files]

    for labelpath, imgpath in zip(labelfiles, imgfiles):
        mosaic_img = load_img(imgpath)
        mosaic = np.load(os.path.join(submission_dir, labelpath))
        plot_multiple_images([mosaic_img, mosaic, image_with_labels(mosaic_img, mosaic)],
                                ['image', 'labels', 'img with labels'],
                                1, 3)



def main():

    if USER == 'antor':
        subs = [
            'submission_DSB2018_512_512_True_12_28_256_0.3_gment_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180406203610_.csv', 
            'submission_DSB2018_512_512_True_12_28_256_0.3_gment_flips_rots_color_balanced_dim_o-tf-horiz-True-rots-True-verti-True_0_80_20180407075023_.csv',
        ]
        compare_submissions([os.path.join(submissions_dir,sub) for sub in subs])
    else:

        # Overwrite filenames with the submissions you wish to compare
        compare_submissions([os.path.join(submissions_dir, 'submission_DSB2018_512_512_True_56_28_256_0.3_gment_double_invert_semantic_dim_o-tf-horiz-True-rots-True-verti-True-zoom_-0.8-1_0.5_25_20180409172401_.csv'),
                             os.path.join(submissions_dir, 'submission_DSB2018_512_512_True_12_28_256_0.3_gment_double_invert_dim_o-tf-horiz-True-rots-True-verti-True-zoom_-0.8-1_0.5_25_20180408211013_.csv')])
        #mosaics_from_submissions('D:/Kaggle/Data_Science_Bowl_2018/data/DSB2018_512_512_True_12_28_256_0.3_gment_2inv_mos_dim_o-tf-horiz-True-rots-True-verti-True_1.0/submission_20180404212329')
        #masks_for_test_mosaics('D:/Kaggle/Data_Science_Bowl_2018/data/DSB2018_512_512_True_12_28_256_0.3_gment_2inv_mos_dim_o-tf-horiz-True-rots-True-verti-True_1.0/submission_20180404233819')

if __name__ == '__main__':
    main()