import os
import argparse

from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf
import numpy as np
import cv2

from utils import constants
from utils import utils


def load_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img) 
    img_tensor = np.expand_dims(img_tensor, axis=0) 
    img_tensor /= 255. 
    return img_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Image path')
    parser.add_argument(
        '--image_filepath', 
        type=str, 
        default=constants.DEFAULT_TEST_IMAGE_PATH,      
        help='Path of the image to be processed'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_filepath = args.image_filepath

    # Load model
    model = tf.keras.models.load_model(
        constants.MODEL_PATH,
        custom_objects={
            'dice_coef': utils.dice_coef, 
            'dice_loss': utils.dice_loss
        },
        compile=False
    )

    # Load the image and predict using the model
    img = load_image(image_filepath, constants.IMAGE_SHAPE_2D)
    predictions = model.predict(img)

    # Save prediction result as png to the results folder
    result_image = predictions[0]
    image_filename = os.path.basename(image_filepath).split('.')[0]
    result_path = os.path.join(
        constants.RESULTS_DIR, 
        f"{image_filename}-prediction.png"
    )
    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(result_path, result_image)

    print(f'[INFO] Saved the prediction result to [{result_path}]')


if __name__ == '__main__':
    main()