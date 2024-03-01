import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from utils import constants
from utils import utils


# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")


class UNet():
    def conv_block(inputs, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)

        return x


    def encoder_block(inputs, num_filters):
        x = UNet.conv_block(inputs, num_filters)
        p = tf.keras.layers.MaxPool2D((2,2))(x)
        return x, p


    def decoder_block(inputs, skip, num_filters):
        x = tf.keras.layers.Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(inputs)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = UNet.conv_block(x, num_filters)
        return x


    def build_unet(input_shape):
        inputs = tf.keras.layers.Input(input_shape)

        # Encoder
        s1, p1 = UNet.encoder_block(inputs, 64) 
        s2, p2 = UNet.encoder_block(p1, 128) 
        s3, p3 = UNet.encoder_block(p2, 256) 
        s4, p4 = UNet.encoder_block(p3, 512)

        # Bridge
        b1 = UNet.conv_block(p4, 1024)

        # Decoder
        d1 = UNet.decoder_block(b1, s4, 512)
        d2 = UNet.decoder_block(d1, s3, 256)
        d3 = UNet.decoder_block(d2, s2, 128)
        d4 = UNet.decoder_block(d3, s1, 64)

        outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        model = tf.keras.models.Model(inputs, outputs, name="UNET")
        return model


    def build_mini_unet(input_shape):
        inputs = tf.keras.layers.Input(input_shape)

        # Encoder
        s1, p1 = UNet.encoder_block(inputs, 16) 
        s2, p2 = UNet.encoder_block(p1, 32) 

        # Bridge
        b1 = UNet.conv_block(p2, 64) 

        # Decoder
        d1 = UNet.decoder_block(b1, s2, 32) 
        d2 = UNet.decoder_block(d1, s1, 16)

        outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d2)
        model = tf.keras.models.Model(inputs, outputs, name="MiniUNET")
        return model


def create_generators(train_data):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_dataframe(
            dataframe=train_data,
            x_col="image_filepath",
            class_mode=None,
            color_mode="rgb",
            target_size=constants.IMAGE_SHAPE_2D, 
            batch_size=constants.BATCH_SIZE,
            seed=42)

    mask_datagen =  ImageDataGenerator()
    mask_generator = mask_datagen.flow_from_dataframe(
            dataframe=train_data,
            x_col="mask_filepath", 
            class_mode=None,
            color_mode="grayscale",
            target_size=constants.IMAGE_SHAPE_2D, 
            batch_size=constants.BATCH_SIZE,
            seed=42) 

    # Combine generators into one which yields image and masks
    def combined_generator(img_gen, mask_gen):
        while True:
            try:
                yield(img_gen.next(), mask_gen.next())
            except OSError: 
                continue
    
    return image_generator, mask_generator, combined_generator


def main():
    # Load data
    train_data = pd.read_csv(constants.TRAIN_METADATA_PATH)
    print(f"[INFO] Loaded [{len(train_data)}] rows of data.")

    # Create generators
    image_generator, mask_generator, combined_generator = create_generators(train_data)
    print("[INFO] Created generators.")

    # Build UNet model
    model = UNet.build_mini_unet(constants.IMAGE_SHAPE_3D)
    print("[INFO] Built UNet model.")

    # Compile the model
    model.compile(optimizer='adam', loss=utils.dice_loss, metrics=[utils.dice_coef])
    print("[INFO] Compiled model.")

    # Use the combined generator to train the model
    steps_per_epoch = len(train_data) // constants.BATCH_SIZE
    model.fit(
        combined_generator(image_generator, mask_generator), 
        steps_per_epoch=steps_per_epoch, 
        epochs=constants.EPOCHS,
    )

    # Save the trained model
    model.save(constants.MODEL_PATH)
    print(f"[INFO] Saved model to [{constants.MODEL_PATH}].")


if __name__ == "__main__":
    main()
