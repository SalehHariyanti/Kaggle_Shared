import numpy as np
np.random.seed(1234)
import pandas as pd

def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def numpy2encoding_no_overlap(predicts, img_name):

    sum_predicts = np.sum(predicts, axis=2)
    sum_predicts[sum_predicts>=2] = 0
    sum_predicts = np.expand_dims(sum_predicts, axis=-1)
    predicts = predicts * sum_predicts
    
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]): 
        rle = run_length_encoding(predicts[:,:,i])
        if len(rle)>0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)    
    return ImageId, EncodedPixels


def numpy2encoding_no_overlap_threshold(predicts, img_name, scores, threshold = 30):

    if predicts.shape[-1] > 0:

        this_threshold = threshold + (threshold * (min(np.product(predicts.shape[:2]), (512 * 512)) - (256 * 256)) / (512 * 512))
        valid = np.sum(predicts, axis = (0, 1)) >= this_threshold
        predicts = predicts[:, :, valid]
        scores = scores[valid]

        sum_predicts = np.sum(predicts, axis=2)
        rows, cols = np.where(sum_predicts>=2)
    
        for i in zip(rows, cols):
            instance_indicies = np.where(np.any(predicts[i[0],i[1],:]))[0]
            highest = instance_indicies[0]
            predicts[i[0],i[1],:] = predicts[i[0],i[1],:]*0
            predicts[i[0],i[1],highest] = 1
    
        ImageId = []
        EncodedPixels = []
        for i in range(predicts.shape[2]): 
            rle = run_length_encoding(predicts[:,:,i])
            if len(rle)>0:
                ImageId.append(img_name)
                EncodedPixels.append(rle)    
    else:
        ImageId = [img_name]
        EncodedPixels = ['']

    if len(ImageId) == 0:
        ImageId = [img_name]
        EncodedPixels = ['']

    return ImageId, EncodedPixels



def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({ 'ImageId' : ImageId , 'EncodedPixels' : EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])

