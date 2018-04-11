
import sys
sys.path.append('../')

import utils
import numpy as np
np.random.seed(1234)
import imageio
import skimage
from scipy import ndimage
import os
from settings import train_dir, data_dir, train_group_id_file, test_group_id_file, supplementary_group_id_file, gan_group_id_file
import pandas as pd
import dsb2018_utils as du
import glob
import math
from enum import Enum

class DSB2018_Dataset(utils.Dataset):
    """Override:
            load_image()
            load_mask()
            image_reference()
    """

    Cache = Enum("Cache",'NONE DISK DISK_MASKS')

    def __init__(self, class_map=None, invert_type = 1, to_grayscale=True, cache=Cache.DISK):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.invert_type = invert_type
        self.to_grayscale = to_grayscale
        self.cache = cache

    def add_nuclei(self, root_dirs, mode, split_ratio=0.9, shuffle = True, target_cluster_id = None, target_maskcount_id = None, target_colour_id = None, use_mosaics=False):
        # Add classes
        self.add_class("nuclei", 1, "nuclei") # source, id, name. id = 0s is BG

        if not isinstance(root_dirs, list):
            root_dirs = [root_dirs]

        image_names = []
        for root_dir in root_dirs:
            if use_mosaics:
                image_names.extend(glob.glob(os.path.join(root_dir, '*.png')))
            else:            
                image_names.extend([os.path.join(root_dir, d) for d in os.listdir(root_dir)])

        image_names.sort()

        if shuffle:
            image_names = list(np.random.permutation(image_names))
        
        length = len(image_names)
        train_item_len = math.floor(split_ratio*length)

        if mode == 'train':
            image_names = image_names[:train_item_len]
        if mode == 'val':
            image_names = image_names[train_item_len:]
        if mode == 'val_as_test':
            image_names = image_names[train_item_len:]     
            mode = 'test'

        if use_mosaics is False:

            dirs = [os.path.join(img_name, 'images') for img_name in image_names]
            mask_dirs = [os.path.join(img_name, 'masks') for img_name in image_names]
            image_ids = [os.path.split(name)[-1] for name in image_names]

            # Get ids
            cluster_id, colour_id, maskcount_id, mosaic_id, mosaic_position = get_ids(image_ids)

            # Strip down to targets only
            idx = np.ones(cluster_id.shape, dtype = np.bool)
            if target_cluster_id is not None:
                idx = np.logical_and(idx, du.ismember(cluster_id, target_cluster_id, index_requested = False))        
            if target_colour_id is not None:
                idx = np.logical_and(idx, du.ismember(colour_id, target_colour_id, index_requested = False))        
            if target_maskcount_id is not None:
                idx = np.logical_and(idx, du.ismember(maskcount_id, target_maskcount_id, index_requested = False))   
            idx = np.argwhere(idx).reshape(-1,)

            # Add images
            for i in idx:
                self.add_image(
                    source = "nuclei", 
                    image_id = i,
                    path = os.path.join(dirs[i], image_ids[i] + '.png'),    
                    mask_dir = mask_dirs[i],
                    name = image_ids[i],
                    cluster_id = cluster_id[i],
                    colour_id = colour_id[i],
                    maskcount_id = maskcount_id[i],
                    mosaic_id = mosaic_id[i],
                    mosaic_position = mosaic_position[i],
                    is_mosaic = False
                    )
        else:
            for i, image_name in enumerate(image_names):
                self.add_image(
                    source = "nuclei", 
                    image_id = i,
                    path = image_name,    
                    name = os.path.split(image_name)[-1],
                    mosaic_id = os.path.split(image_name)[-1].split('.')[0],
                    is_mosaic = True
                    )
      
    def get_cache_dir(self, is_mask):
        return os.path.join(data_dir, '_'.join(('maskrcnn_mask_cache' if is_mask else 'maskrcnn_image_cache', str(self.invert_type), str(self.to_grayscale))))

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = self.load_image_from_file(image_id)
        return image

    def load_image_from_file(self, image_id):

        if self.image_info[image_id]['is_mosaic'] is False:

            image = None

            if self.cache == DSB2018_Dataset.Cache.DISK:

                image_file = os.path.join(self.get_cache_dir(False), ''.join((self.image_info[image_id]['name'], '.npy')))

                if not os.path.exists(self.get_cache_dir(False)):
                    os.makedirs(self.get_cache_dir(False))

                if os.path.exists(image_file):
                    image = np.load(image_file)

            if image is None:

                image = imageio.imread(self.image_info[image_id]['path'])
                # RGBA to RGB
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis = -1)
                elif image.shape[2] != 3:
                    image = image[:,:,:3]
                
                # Grey and invert
                if self.to_grayscale:
                    image = skimage.color.rgb2gray(image.astype('uint8'))
                    # NB: skimage.color.rgb2gray converts automatically to float 0-1 scale!!!!
                    image = (image * 255).astype(np.uint8)
                    if self.invert_type == 1:
                        image = self.invert_img(image)
                    elif self.invert_type == 2:
                        image = self.invert_img(self.invert_img(image), cutoff = -1)
                    image = np.stack([image] * 3, axis = -1)

                if self.cache == DSB2018_Dataset.Cache.DISK:
                    np.save(image_file, image)

        else:

            image = imageio.imread(self.image_info[image_id]['path'])
            # RGBA to RGB
            if image.ndim == 2:
                image = np.stack([image] * 3, axis = -1)
            elif image.shape[2] != 3:
                image = image[:,:,:3]

            # Grey and invert
            if self.to_grayscale:
                image = skimage.color.rgb2gray(image.astype('uint8'))
                # NB: skimage.color.rgb2gray converts automatically to float 0-1 scale!!!!
                image = (image * 255).astype(np.uint8)
                if self.invert_type == 1:
                    image = self.invert_img(image)
                elif self.invert_type == 2:
                    image = self.invert_img(self.invert_img(image), cutoff = -1)
                image = np.stack([image] * 3, axis = -1)

        return image
    
    def image_reference(self, image_id):
        """Return the details of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["path"]
        else:
            super(DSB2018_Dataset, self).image_reference(self, image_id)

    def load_mask(self, image_id):
        """ 
        Returns:
            masks: A binary array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        mask = self.load_mask_from_file(image_id)

        class_ids = np.ones(mask.shape[-1], dtype=np.int32)

        return mask, class_ids

    def load_mask_from_file(self, image_id):

        if self.image_info[image_id]['is_mosaic'] is False:

            mask = None

            if (self.cache == DSB2018_Dataset.Cache.DISK) or (self.cache == DSB2018_Dataset.Cache.DISK_MASKS):

                mask_file = os.path.join(self.get_cache_dir(True), ''.join((self.image_info[image_id]['name'], '.npy')))

                if not os.path.exists(self.get_cache_dir(True)):
                    os.makedirs(self.get_cache_dir(True))

                if os.path.exists(mask_file):
                    mask = np.load(mask_file)

            if mask is None:

                info = self.image_info[image_id]
                mask_dir = info['mask_dir'] 
                mask_names = os.listdir(mask_dir)
                mask_paths = [os.path.join(mask_dir, mask_name) for mask_name in mask_names]
            
                count = len(mask_paths)
            
                masks = [self.fill_img(imageio.imread(path)) for path in mask_paths]
                mask = np.stack(masks, axis=-1)

                if (self.cache == DSB2018_Dataset.Cache.DISK) or (self.cache == DSB2018_Dataset.Cache.DISK_MASKS):
                    np.save(mask_file, mask)

        else:

            mask_file = self.image_info[image_id]['path'][:-4] + '.npz'

            mask_mosaic_load = np.load(mask_file)

            mask = mask_mosaic_load['mask_mosaic']

        return mask   

    def invert_img(self, img, cutoff=.5):
        '''Invert image if mean value is greater than cutoff.'''

        img_dtype = img.dtype

        # Normalise and Invert
        img = img / img.max()
        if np.mean(img) > cutoff:
            img = 1 - img
       
        # Convert back to original dtype
        if img_dtype == np.uint8:
            img = (img * 255).astype(np.uint8)

        return img

    def fill_img(self, img):
        return ndimage.binary_fill_holes(img).astype(np.uint8)


def get_ids(file_id):
    """
    Returns id based on mosaic membership
    (see clustering_functions.py)
    """
    assert os.path.exists(train_group_id_file)
    assert os.path.exists(supplementary_group_id_file)
    assert os.path.exists(test_group_id_file)
    assert os.path.exists(gan_group_id_file)

    train_df = pd.read_csv(train_group_id_file)
    supplementary_df = pd.read_csv(supplementary_group_id_file)
    test_df = pd.read_csv(test_group_id_file)
    gan_df = pd.read_csv(gan_group_id_file)

    mosaic_df = pd.concat([train_df, supplementary_df, test_df, gan_df])

    mosaic_file_id = np.array(mosaic_df['img_id'])
    mosaic_idx = np.array(mosaic_df['mosaic_idx'])
    mosaic_position = np.array(mosaic_df['mosaic_position'])
    cluster_id = np.array(mosaic_df['cluster_id'])
    colour_id = np.array(mosaic_df['colour_id'])
    maskcount_id = np.array(mosaic_df['maskcount_id'])

    file_id = np.array(file_id)

    A, B = du.ismember(file_id, mosaic_file_id)

    # Assign a mosaic id to each file.
    # In cases where no mosaic id assign it a single value.
    mosaic_id = np.zeros(file_id.shape, dtype = np.int)
    mosaic_pos = np.empty(file_id.shape, dtype = object)
    cluster = np.zeros(file_id.shape, dtype = np.int)
    colour = np.zeros(file_id.shape, dtype = np.int)
    maskcount = np.zeros(file_id.shape, dtype = np.int)

    mosaic_id[A] = mosaic_idx[B[A]]
    colour[A] = colour_id[B[A]] 
    cluster[A] = cluster_id[B[A]]
    maskcount[A] = maskcount_id[B[A]]
    mosaic_pos[A] = mosaic_position[B[A]]

    if np.any(np.logical_not(A)):
        mosaic_id[np.logical_not(A)] = np.cumsum(np.ones((sum(np.logical_not(A)),))) + mosaic_id[A].max()
        cluster[np.logical_not(A)] = np.zeros((sum(np.logical_not(A)),))
        colour[np.logical_not(A)] = np.zeros((sum(np.logical_not(A)),))
        maskcount[np.logical_not(A)] = np.zeros((sum(np.logical_not(A)),))

    return cluster, colour, maskcount, mosaic_id, mosaic_pos
def main():
    return

if __name__ == "__main__":
    main()