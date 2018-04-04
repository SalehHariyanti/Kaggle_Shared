import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv
from visualize import *
from dsb2018_utils import * 

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(base_dir, 'test')
submissions_dir = os.path.join(base_dir, 'submissions')

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


def main():
    # Overwrite filenames with the submissions you wish to compare
    compare_submissions([os.path.join(submissions_dir, 'submission_scaled_DSB2018_512_512_True_12_28_256_0.3_caled_2inv_dim_o-tf-horiz-True-verti-True_0.5_None_20180404153552_.csv'),
                         os.path.join(submissions_dir, 'submission_DSB2018_512_512_True_12_28_256_0.3_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180402101810_.csv')])

if __name__ == '__main__':
    main()