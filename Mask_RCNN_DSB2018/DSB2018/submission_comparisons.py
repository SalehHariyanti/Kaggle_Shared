import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import csv

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

data_dir = os.path.join(base_dir, 'data')
test_dir = os.path.join(base_dir, 'test')
submissions_dir = os.path.join(base_dir, 'submissions')

#############
# Comparing submission outputs

def load_img(filename, hsv = False, greyscale = False, adjust_contrast = False, lab = False):


    img = np.array(PIL.Image.open(filename), dtype=np.uint8) if filename.endswith('gif') else cv2.imread(filename)
    
    if img is None:
        print(' '.join((filename, 'corrupted or does not exist.')))
        return None

    # Force three dims if grey (gets converted later if greyscale is requested)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis = -1)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 returns in BGR order

    if hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if adjust_contrast:
        img = adjust_img_contrast(img)

    if lab:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    return img


def resize_img(img, rows, cols):
    """
    Resize an image with cv2
    To shrink an image, it will generally look best with cv::INTER_AREA interpolation, whereas to enlarge an image, it will generally look best with cv::INTER_CUBIC (slow) or cv::INTER_LINEAR (faster but still looks OK).
    """
    interpolation = cv2.INTER_AREA if np.product(img.shape[:2]) > (rows * cols) else cv2.INTER_LINEAR
    return cv2.resize(img, (cols, rows), interpolation = interpolation)


def grays_to_RGB(img):
    """
    turn 2D grayscale image into grayscale RGB
    """
    return np.dstack((img, img, img)) 


def image_with_masks(img, masks, edge_colour = 255):
    """
    returns a copy of the image with edges of the masks added 
    (up to three masks can be added)
    """

    img_color = grays_to_RGB(img.copy()) if img.ndim < 3 else img.copy()
    img_color = (img_color * 255).astype(np.uint8) if np.max(img_color) <= 1 else img_color
    mask_edges = [cv2.Canny((masks[i] * 255).astype(np.uint8), 100, 200) > 0 for i in range(min(3, len(masks)))]

    for i in range(len(mask_edges)):
        img_color[mask_edges[i], i] = edge_colour 

    return img_color


def image_with_labels(img, label, edge_colour = 255):
    """
    returns a copy of the image with edges of the masks added 
    (up to three masks can be added)
    """

    img_color = grays_to_RGB(img.copy()) if img.ndim < 3 else img.copy()
    img_color = (img_color * 255).astype(np.uint8) if np.max(img_color) <= 1 else img_color

    u_label = np.unique(label)
    mask_edges = [cv2.Canny(((label == i) * 255).astype(np.uint8), 100, 200) > 0 for i in u_label[u_label != 0]]
    mask_edges = np.max(np.array(mask_edges), axis = 0)

    for i in range(3):
        img_color[mask_edges, i] = edge_colour 

    return img_color


def plot_multiple_images(img_list, title_list = None, nrows = 3, ncols = 6, greyscale = False):

    title_list = ['' for i in img_list] if title_list is None else title_list
    
    nrows = int(np.ceil(len(img_list) / ncols)) if nrows is None else nrows
    ncols = int(np.ceil(len(img_list) / nrows)) if ncols is None else ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, img, t in zip(axes.flat, img_list, title_list):
        if greyscale:
            ax.imshow(img.astype(np.uint8), cmap = 'gray')
        else:
            ax.imshow(img.astype(np.uint8))
        ax.set_title(t)

    plt.show()

    return


def run_length_decode(rel, H, W, fill_value = 255, index_offset = 0):
    mask = np.zeros((H * W), np.uint8)
    if rel != '':
        rel  = np.array([int(s) for s in rel.split(' ')]).reshape(-1, 2)
        for r in rel:
            start = r[0] - index_offset
            end   = start + r[1]
            mask[start : end] = fill_value
    mask = mask.reshape(H, W)
    return mask


def labels_from_rles(mask_rles, mask_shape):

    masks = [run_length_decode(rle, mask_shape[1], mask_shape[0], 1, index_offset = 1).T for rle in mask_rles]
    labels = np.zeros(mask_shape, dtype = np.int)

    for i, mask in enumerate(masks):
        labels += (mask * (i + 1))

    return labels, masks


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
        test_img = load_img(os.path.join(test_dir, this_file, 'images', ''.join((this_file, '.png'))))
        labels = [labels_from_rles(sr[np.argwhere(sf == this_file).reshape(-1,)][0], test_img.shape[:2])[0] for sr, sf in zip(submissions_rles, submissions_filenames)]
        plot_multiple_images([test_img] + [image_with_labels(test_img, l) for l in labels] + [image_with_masks(test_img, labels)], ['img'] + ['_'.join(('submission', str(i), str(np.max(l)))) for i, l in enumerate(labels)] + ['img_with_masks'], 1, len(labels) + 2)

    return


compare_submissions([os.path.join(submissions_dir, 'submission_DSB2018_512_512_True_12_28_256_0.3_double_invert_dim_o-tf-horiz-True-verti-True_0.5_None_20180402101810_.csv'),
                     os.path.join(submissions_dir, 'submission_DSB2018_512_512_True_12_28_256__dim_o-tf-horiz-True-verti-True_0.5_None_20180326101255_.csv')])