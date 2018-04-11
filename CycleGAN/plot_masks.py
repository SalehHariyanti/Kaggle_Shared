import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import skimage.io
import os


STAGE1_TRAIN               = "stage1_train"
STAGE1_TRAIN_IMAGE_PATTERN = "%s/{}/images/{}.png" % STAGE1_TRAIN
STAGE1_TRAIN_MASK_PATTERN  = "%s/{}/masks/*.png" % STAGE1_TRAIN

GAN1                       = "black2purple_aug"
GAN2                       = "black2nbs_aug"
GAN1_IMAGE_PATTERN         = "%s/{}_GAN1.png" % GAN1
GAN2_IMAGE_PATTERN         = "%s/{}_GAN2.png" % GAN2


# Get image width, height and count masks available.
def read_image_labels(image_id, space="rgb"):
    # Image
    image_file = STAGE1_TRAIN_IMAGE_PATTERN.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    image = image[:, :, :3]     # Drop alpha which is not used

    # GAN1 image
    gan1_file = GAN1_IMAGE_PATTERN.format(image_id)
    gan1 = skimage.io.imread(gan1_file)

    # GAN2 image
    gan2_file = GAN2_IMAGE_PATTERN.format(image_id)
    gan2 = skimage.io.imread(gan2_file)

    # Mask
    mask_file = STAGE1_TRAIN_MASK_PATTERN.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()

    height, width, _ = image.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = 255 #index + 1
    return image, gan1, gan2, labels, num_masks



def plot_image_masks(image, gan1, gan2, labels, num_masks, image_id):
    fig, ax = plt.subplots(2,3,figsize=(16,5))

    plt.suptitle(image_id)

    # Plot image
    d = ax[0][0].axis('off')
    d = ax[0][0].imshow(image)
    d = ax[0][0].set_title("image")

    # Plot gan1
    d = ax[0][1].axis('off')
    d = ax[0][1].imshow(gan1)
    d = ax[0][1].set_title("Gan1")

    # Plot gan2
    d = ax[0][2].axis('off')
    d = ax[0][2].imshow(gan2)
    d = ax[0][2].set_title("Gan2")

    # Plot img + mask
    d = ax[1][0].axis('off')
    d = ax[1][0].imshow(image)
    d = ax[1][0].imshow(labels, alpha=0.3)
    d = ax[1][0].set_title("masks: %d"%num_masks)

    # Plot gan1 + mask
    d = ax[1][1].axis('off')
    d = ax[1][1].imshow(gan1)
    d = ax[1][1].imshow(labels, alpha=0.3)
    d = ax[1][1].set_title("Gan1 + masks")

    # Plot gan2 + mask
    d = ax[1][2].axis('off')
    d = ax[1][2].imshow(gan2)
    d = ax[1][2].imshow(labels, alpha=0.3)
    d = ax[1][2].set_title("Gan2 + masks")

    plt.show()
    #input("Press any key...")
    #plt.close()


def display_image_masks(image_id):
    image, gan1, gan2, labels, num_masks = read_image_labels(image_id)
    plot_image_masks(image, gan1, gan2, labels, num_masks, image_id)


df = pd.read_csv('train_df.csv')
ids = df[df["HSV_CLUSTER"]==0]["img_id"].tolist()
print(len(ids))

for id in ids:
    display_image_masks(id)