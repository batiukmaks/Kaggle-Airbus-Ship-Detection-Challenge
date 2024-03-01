import os
import shutil

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2

from utils import constants
from utils import utils


def copy_images(df):
    """Copy images to their locations"""
    for _, row in df.iterrows():
        source = os.path.join(constants.RAW_TRAIN_IMAGES_DIR, row["image_id"])
        destination = row["image_filepath"]
        destination_dir = os.path.dirname(destination)

        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy(source, destination)


def save_masks(df):
    """Save masks to their locations"""
    for _, row in df.iterrows():
        mask = row["mask"]
        destination = row["mask_filepath"]
        destination_dir = os.path.dirname(destination)

        os.makedirs(destination_dir, exist_ok=True)
        if row["ship_located"]:
            cv2.imwrite(destination, mask)
        else:
            cv2.imwrite(
                os.path.join(destination), 
                np.zeros(constants.IMAGE_SHAPE_2D, dtype=np.uint8)
            )


def main():
    # Load the data
    raw_df = pd.read_csv(constants.RAW_METADATA_PATH, na_values=[np.nan, ""])
    segmentations_df = raw_df.rename(
        columns={
            "ImageId": "image_id", 
            "EncodedPixels": "encoded_pixels"
        }
    )
    segmentations_df["ship_located"] = segmentations_df["encoded_pixels"].notnull()
    print(f"[INFO] Read [{len(segmentations_df)}] rows of data.")

    # Combine the masks for the same image
    segmentations_df["mask"] = segmentations_df["encoded_pixels"].apply(
        lambda x: utils.rle_to_mask(x, shape=constants.INITIAL_MASK_SHAPE) if pd.notna(x) else x
    )
    grouped_df = (
        segmentations_df.groupby("image_id")
        .agg({"mask": lambda x: np.max(np.stack(x.values), axis=0)})
        .reset_index()
    )

    # Split the data into train and test sets
    grouped_df["ship_located"] = grouped_df["mask"].notnull()
    train_df, test_df = train_test_split(
        grouped_df,
        test_size=0.2,
        stratify=grouped_df["ship_located"],
        random_state=42
    )
    print(f"[INFO] Sets have [{train_df['ship_located'].mean() * 100:.3f}%] images with ships.")

    # Compose filepaths
    train_df["image_filepath"] = train_df["image_id"].apply(
        lambda x: os.path.join(constants.TRAIN_IMAGES_DIR, x)
    )
    test_df["image_filepath"] = test_df["image_id"].apply(
        lambda x: os.path.join(constants.TEST_IMAGES_DIR, x)
    )

    train_df["mask_filepath"] = train_df["image_id"].apply(
        lambda x: os.path.join(
            constants.TRAIN_MASKS_DIR, 
            x.replace(".jpg", ".png")
        )
    )
    test_df["mask_filepath"] = test_df["image_id"].apply(
        lambda x: os.path.join(
            constants.TEST_MASKS_DIR, 
            x.replace(".jpg", ".png")
        )
    )
    print(f"[INFO] Composed filepaths for images and masks.")

    # Copy images and save masks
    copy_images(train_df)
    copy_images(test_df)
    print(f"[INFO] Copied images to their locations.")

    save_masks(train_df)
    save_masks(test_df)
    print(f"[INFO] Saved masks to their locations.")

    # Save the train and test dataframes to csv
    train_df.to_csv(constants.TRAIN_METADATA_PATH, index=False)
    print(f"[INFO] Saved train metadata to [{constants.TRAIN_METADATA_PATH}]")
    test_df.to_csv(constants.TEST_METADATA_PATH, index=False)
    print(f"[INFO] Saved test metadata to [{constants.TEST_METADATA_PATH}]")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()