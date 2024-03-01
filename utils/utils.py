import numpy as np
import pandas as pd
from keras import backend as K


from utils import constants


def rle_to_mask(rle_list, shape):
    """Convert RLE string to mask"""
    mask = np.zeros(shape).astype(np.uint8)

    if pd.isna(rle_list) or len(rle_list) == 1:
        return mask
    else:
        # Split the RLE string to get start and length of the mask
        rle_list = [int(i) for i in rle_list.split()]
        # Get the pixel positions
        pixels = [
            (pixel_position % shape[1] - 1, pixel_position // shape[1] - 1)
            for start, length in list(zip(rle_list[0:-1:2], rle_list[1::2]))
            for pixel_position in range(start, start + length)
        ]
        for pixel in pixels:
            mask[pixel[1], pixel[0]] = 255
    return mask


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
