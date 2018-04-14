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
import PIL
from PIL import Image

import getpass
USER = getpass.getuser()

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018' if os.name == 'nt' else os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(base_dir, 'test')

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


def extract_submission(submission_file):

    submissions_data = extract_data(submission_file)

    submissions_filenames = submissions_data[0]
    submissions_rles = submissions_data[1]

    for i in range(len(submissions_filenames)):

        this_file = submissions_filenames[i]
        test_img = load_img(os.path.join(test_dir, this_file, 'images', ''.join((this_file, '.png'))), greyscale = True)
        labels, masks = labels_from_rles(submissions_rles[np.argwhere(submissions_filenames == this_file).reshape(-1,)][0], test_img.shape)

        # Plot
        if False:
            plot_multiple_images([test_img] + [image_with_labels(test_img, labels)])

        # Save them
        mask_filepaths = [os.path.join(test_dir, this_file, 'masks', ''.join((this_file, '_', str(i), '.png'))) for i in range(len(masks))]
        for mask_filepath, mask in zip(mask_filepaths, masks):
            mask = Image.fromarray(mask * 255)
            if not os.path.exists(os.path.split(mask_filepath)[0]):
                os.makedirs(os.path.split(mask_filepath)[0])                                  
            mask.save(mask_filepath)
    return



def main():
    extract_submission(os.path.join(data_dir, 'stage1_solution.csv'))
                             

if __name__ == '__main__':
    main()