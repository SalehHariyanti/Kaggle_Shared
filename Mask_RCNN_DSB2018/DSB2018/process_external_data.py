import os
import numpy as np
import shutil
import PIL
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from bs4 import BeautifulSoup
import cv2
from shutil import copyfile

base_dir = 'D:/Kaggle/Data_Science_Bowl_2018'

#os.path.join(base_dir, 'external_data', 'ISBI', 'TrainTestAnnotations', 'Dataset', 'EDF') =>  https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/dataset.html 
#os.path.join(base_dir, 'external_data', 'nuclei_segmentation_benchmark') => https://nucleisegmentationbenchmark.weebly.com/


def process_isbi(dir = os.path.join(base_dir, 'external_data', 'ISBI', 'TrainTestAnnotations', 'Dataset', 'EDF')):

    new_dir = os.path.join(base_dir, 'train_external', 'ISBI')

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    file_ids = [os.path.splitext(f)[0] for f in os.listdir(dir) if '_GT' not in f]

    file_ids_fill = ['EDF000', 'EDF001', 'EDF002']

    for file in file_ids:

        original_filename = os.path.join(dir, ''.join((file, '.png')))
        mask_filename = os.path.join(dir, ''.join((file, '_GT.png')))

        # copy original
        img = Image.open(original_filename)
        img.convert("RGB")
        if not os.path.exists(os.path.join(new_dir, file, 'images')):
            os.makedirs(os.path.join(new_dir, file, 'images'))
        img.save(os.path.join(new_dir, file, 'images', ''.join((file, '.png'))))

        # load mask and reformat
        #masks = np.array(Image.open(mask_filename), dtype = np.uint8)
        masks = load_img(mask_filename)[:, :, 0]
        if file in file_ids_fill:
            masks = fill_img(masks) 
        masks = mask_to_multi(masks)

        # save individual masks as png
        mask_filepaths = [os.path.join(new_dir, file, 'masks', ''.join((file, '_', str(i), '.png'))) for i in range(len(masks))]

        for mask_filepath, mask in zip(mask_filepaths, masks):

            mask = Image.fromarray(mask)

            if not os.path.exists(os.path.split(mask_filepath)[0]):
                os.makedirs(os.path.split(mask_filepath)[0])
                                  
            mask.save(mask_filepath)
      
  
    return


def process_nuclei_segmentation_benchmark(dir = os.path.join(base_dir, 'external_data', 'nuclei_segmentation_benchmark'), n_crops = 8):

    file_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(dir, 'Annotations'))]

    mask_dir = os.path.join(dir, 'Masks')

    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # Create masks from annotations
    for f in file_ids:

        mask_subdir = os.path.join(dir, 'Masks', f)
        if not os.path.exists(mask_subdir):
            os.makedirs(mask_subdir)

        img = load_img(os.path.join(dir, 'Tissue images', '.'.join((f, 'tif'))))
        if img is None:
            img = load_img(os.path.join(dir, 'Tissue images', '.'.join((f, 'tiff'))))
          
        file = open(os.path.join(dir, 'Annotations', '.'.join((f, 'xml')))).read()
           
        soup = BeautifulSoup(file, 'xml')

        regions = soup.find_all('Region')

        n_regions = len(regions)
        contours = [[]] * n_regions

        for i, region in enumerate(regions):
            for vertices in region.find_all('Vertices', recursive = False):
                contours[i] = np.array([np.array([float(vertex.attrs['X']), float(vertex.attrs['Y'])]).reshape(1, 2) 
                                        for vertex in vertices.find_all('Vertex', recursive = False)])
 
        for i, contour in enumerate(contours):

            mask_filepath = os.path.join(mask_subdir, ''.join((f, '_', str(i), '.png')))
            
            if not os.path.isfile(mask_filepath):
            
                mask = np.zeros(img.shape[:2], dtype = np.uint8)
                mask = cv2.drawContours(mask, [contour.astype(np.int32)], 0, 255, -1)

                if False:
                    import visualisation as vis
                    vis.plot_multiple_images([img, mask, vis.image_with_masks(img, [mask])], ['img', 'mask', 'img_with_mask'], 1, 3)

                # Save the mask
                mask = Image.fromarray(mask)
                mask.save(mask_filepath)


    # Save originals to new training dirs
    new_dir = os.path.join(base_dir, 'train_external', 'nsb')

    for f in file_ids:

        if not os.path.exists(os.path.join(new_dir, f, 'images')):
            os.makedirs(os.path.join(new_dir, f, 'images'))
        if not os.path.exists(os.path.join(new_dir, f, 'masks')):
            os.makedirs(os.path.join(new_dir, f, 'masks'))
        
        # Load the original mask and save as a png file
        img = load_img(os.path.join(dir, 'Tissue images', '.'.join((f, 'tif'))))
        if img is None:
            img = load_img(os.path.join(dir, 'Tissue images', '.'.join((f, 'tiff'))))
        img = Image.fromarray(img)
        img.save(os.path.join(new_dir, f, 'images', '.'.join((f, 'png'))))

        # Copy the masks
        for mask in os.listdir(os.path.join(mask_dir, f)):
            copyfile(os.path.join(mask_dir, f, mask), os.path.join(new_dir, f, 'masks', mask))


    # Crops from same set
    crops_dir = os.path.join(base_dir, 'train_external', '_'.join(('nsb', 'crop')))
    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir)

    for f in file_ids:

        img = load_img(os.path.join(new_dir, f, 'images', '.'.join((f, 'png'))))
        mask_filenames = [os.path.join(new_dir, f, 'masks', file) for file in os.listdir(os.path.join(new_dir, f, 'masks'))]
        masks = [load_img(file)[:, :, 0] for file in mask_filenames]
        all_masks = (np.sum(np.dstack(masks), axis = -1) > 0).astype(np.int)
        all_crops = all_masks * 0

        for crop in range(n_crops):

            crop_file = '_'.join((f, 'crop', str(crop)))

            index_found = False
            counter = 0
            while not index_found:
                counter += 1
                rnd_mask, crop_index = random_crop(all_masks, (np.random.randint(256, 512), np.random.randint(256, 512)))
             
                new_crop = np.zeros_like(all_crops)
                new_crop[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3]] = 1

                # Only take the crop if it covers 5% of the masks in the image and if it doesn't overlap more than 30% with crops 
                # that we have already taken
                if ((np.sum(rnd_mask) > np.sum(all_masks > 0) * 0.05) and (np.sum(new_crop * all_crops) < 0.3 * np.sum(new_crop))):
                    index_found = True
                    all_crops[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3]] = 1
                elif counter == 100:
                    index_found = True
                    
            # import visualisation as vis
            # vis.plot_multiple_images([img[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3], :], masks[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3]], vis.image_with_masks(img[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3], :], [masks[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3]]])], ['img', 'masks', 'img_masks'], 1, 3)
            if counter < 100:
                
                if not os.path.exists(os.path.join(crops_dir, crop_file, 'images')):
                    os.makedirs(os.path.join(crops_dir, crop_file, 'images'))

                rnd_img = Image.fromarray(img[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3], :])
                rnd_img.save(os.path.join(crops_dir, crop_file, 'images', ''.join((crop_file, '.png'))))

                rnd_mask = [m[crop_index[0] : crop_index[2], crop_index[1] : crop_index[3]] for m in masks]
                rnd_mask = [m for m in rnd_mask if np.sum(m) > 0]

                # save individual masks as png
                mask_filepaths = [os.path.join(crops_dir, crop_file, 'masks', ''.join((crop_file, '_', str(i), '.png'))) for i in range(len(rnd_mask))]

                for mask_filepath, mask in zip(mask_filepaths, rnd_mask):

                    mask = Image.fromarray(mask)

                    if not os.path.exists(os.path.split(mask_filepath)[0]):
                        os.makedirs(os.path.split(mask_filepath)[0])
                                  
                    mask.save(mask_filepath)
        
    
def load_img(filename, greyscale = False):


    img = np.array(PIL.Image.open(filename), dtype=np.uint8) if filename.endswith('gif') else cv2.imread(filename)
    
    if img is None:
        print(' '.join((filename, 'corrupted or does not exist.')))
        return None

    # Force three dims if grey (gets converted later if greyscale is requested)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis = -1)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 returns in BGR order

    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def mask_to_multi(masks):
    lab_mask, number_of_labs = ndimage.label(masks > 0)
    return [((lab_mask == i) * 255).astype(np.uint8) for i in range(1, lab_mask.max() + 1)]


def fill_img(img):
    return ndimage.binary_fill_holes(img).astype(np.uint8)


def random_crop(x, random_crop_size):

    w, h = x.shape[0], x.shape[1]

    if (w, h) != random_crop_size:

        rangew = (w - random_crop_size[0]) // 2
        rangeh = (h - random_crop_size[1]) // 2
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)

        return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]], (offsetw, offseth, offsetw+random_crop_size[0], offseth+random_crop_size[1])

    else:

        return x, (0, 0) + random_crop_size


def main():
    process_isbi()
    process_nuclei_segmentation_benchmark()

if __name__ == '__main__':
    main()
    