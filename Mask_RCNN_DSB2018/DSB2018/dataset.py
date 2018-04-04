
import sys
sys.path.append('../')

import utils
import numpy as np
np.random.seed(1234)
import imageio
import skimage
from scipy import ndimage
import os
from settings import train_dir, data_dir, train_group_id_file, test_group_id_file, supplementary_group_id_file
import pandas as pd
import dsb2018_utils as du
import glob
import math

class DSB2018_Dataset(utils.Dataset):
    """Override:
            load_image()
            load_mask()
            image_reference()
    """

    def __init__(self, class_map=None, invert_type = 1):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        self.invert_type = invert_type
        
    def add_nuclei(self, root_dirs, mode, split_ratio=0.9, shuffle = True, target_colour_id = None, use_mosaics=False):
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
            colour_id, group_id = get_ids(image_ids)

            # Strip down to target colours only
            if target_colour_id is not None:
                idx = du.ismember(colour_id, target_colour_id, index_requested = False)
                idx = np.argwhere(idx).reshape(-1, )
            else:
                idx = np.arange(len(image_names))

            # Add images
            for i in idx:
                self.add_image(
                    source = "nuclei", 
                    image_id = i,
                    path = os.path.join(dirs[i], image_ids[i] + '.png'),    
                    mask_dir = mask_dirs[i],
                    name = image_ids[i],
                    colour_id = colour_id[i],
                    is_mosaic = False
                    )
        else:
            for i, image_name in enumerate(image_names):
                self.add_image(
                    source = "nuclei", 
                    image_id = i,
                    path = image_name,    
                    name = os.path.split(image_name)[-1],
                    is_mosaic = True
                    )

      
    def get_cache_dir(self, is_mask):
        return os.path.join(data_dir, '_'.join(('maskrcnn_mask_cache' if is_mask else 'maskrcnn_image_cache', str(self.invert_type))))

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        image = self.load_image_from_file(image_id)
        return image

    def load_image_from_file(self, image_id):

        if self.image_info[image_id]['is_mosaic'] is False:

            image_file = os.path.join(self.get_cache_dir(False), ''.join((self.image_info[image_id]['name'], '.npy')))

            if not os.path.exists(self.get_cache_dir(False)):
                os.makedirs(self.get_cache_dir(False))

            if os.path.exists(image_file):

                image = np.load(image_file)

            else:
                
                image = imageio.imread(self.image_info[image_id]['path'])
                # RGBA to RGB
                if image.ndim == 2:
                    image = np.stack([image] * 3, axis = -1)
                elif image.shape[2] != 3:
                    image = image[:,:,:3]
                # Grey and invert
                image = skimage.color.rgb2gray(image.astype('uint8'))
                if self.invert_type == 1:
                    image = self.invert_img(image)
                elif self.invert_type == 2:
                    image = self.invert_img(self.invert_img(image), cutoff = -1)
                image = np.stack([image] * 3, axis = -1)

                np.save(image_file, image)

        else:

            image = imageio.imread(self.image_info[image_id]['path'])
            # RGBA to RGB
            if image.ndim == 2:
                image = np.stack([image] * 3, axis = -1)
            elif image.shape[2] != 3:
                image = image[:,:,:3]
            # Grey and invert
            image = skimage.color.rgb2gray(image.astype('uint8'))
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

            mask_file = os.path.join(self.get_cache_dir(True), ''.join((self.image_info[image_id]['name'], '.npy')))

            if not os.path.exists(self.get_cache_dir(True)):
                os.makedirs(self.get_cache_dir(True))

            if os.path.exists(mask_file):

                mask = np.load(mask_file)

            else:
      
                info = self.image_info[image_id]
                mask_dir = info['mask_dir'] 
                mask_names = os.listdir(mask_dir)
                mask_paths = [os.path.join(mask_dir, mask_name) for mask_name in mask_names]
            
                count = len(mask_paths)
            
                masks = [self.fill_img(imageio.imread(path)) for path in mask_paths]
                mask = np.stack(masks, axis=-1)

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

    train_df = pd.read_csv(train_group_id_file)
    supplementary_df = pd.read_csv(supplementary_group_id_file)
    test_df = pd.read_csv(test_group_id_file)

    mosaic_df = pd.concat([train_df, supplementary_df, test_df])

    mosaic_file_id = np.array(mosaic_df['img_id'])
    mosaic_id = np.array(mosaic_df['mosaic_idx'])
    target_id = np.array(mosaic_df['target_id'])

    file_id = np.array(file_id)

    A, B = du.ismember(file_id, mosaic_file_id)

    # Assign a mosaic id to each file.
    # In cases where no mosaic id assign it a single value.
    group_id = np.zeros(file_id.shape, dtype = np.int)
    target = np.zeros(file_id.shape, dtype = np.int)

    group_id[A] = mosaic_id[B[A]]
    target[A] = target_id[B[A]]

    if np.any(np.logical_not(A)):
        group_id[np.logical_not(A)] = np.cumsum(np.ones((sum(np.logical_not(A)),))) + group_id[A].max()
        target[np.logical_not(A)] = np.zeros((sum(np.logical_not(A)),))

    group_id = group_id + 1 # Start at 1 rather than zero

    return target, group_id

def main():
    return

if __name__ == "__main__":
    main()