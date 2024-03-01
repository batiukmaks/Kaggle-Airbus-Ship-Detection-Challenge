# Airbus Ship Detection Challenge
## Description
> Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

> Airbus offers comprehensive maritime monitoring services by building a meaningful solution for wide coverage, fine details, intensive monitoring, premium reactivity and interpretation response. Combining its proprietary-data with highly-trained analysts, they help to support the maritime industry to increase knowledge, anticipate threats, trigger alerts, and improve efficiency at sea.

## Solution Structure
The solution consists of several steps, they have to be performed sequentially.

#### Step 0. Prepare Environment
- Create a Python virtual environment using terminal command `python -m venv venv`.
- Activate it with `source venv/bin/activate`.
    > [!NOTE]
    > It may wary depending on your OS, check [this](https://docs.python.org/3/tutorial/venv.html).
- Then install all the necessary packages using `pip install -r requirements.txt`.
- Set PYTHONPATH environment variable using `export PYTHONPATH=$(pwd)`.


#### Step 1. Download Dataset
- You have to download data from [Kaggle](https://www.kaggle.com/competitions/airbus-ship-detection) using `kaggle competitions download -c airbus-ship-detection` command.

    > [!NOTE]
    > If you are not authorized or the package is not installed, follow the [link](https://github.com/Kaggle/kaggle-api) and follow the steps.

- After .zip file is installed, extract it.

#### Step 2. Set Constants
- In the file `utils/constants.py` you should set the locations of your extracted data.
- The training parameters and others are set in this file too.


#### Step 3. Explore Data (Optional)
- In the `explore.ipynb` the main analysis is presented.
- You can change it if you need to explore data in more detail.


#### Step 4. Split Data
- In this step, the images are split into train and test sets. 
- The annotated segments of ships are combined for each image and saved in PNG format.
- Metadata files are created with images' and masks' locations.
- Run `./pre-processing/split_data.py` to proceed with this step.


#### Step 5. Train Model
- This step involves training your model on the data and parameters you set in the `utils/constants.py`.
- The training is performed using TensorFlow, UNet architecture for the Neural Network, and Dice Score as a loss function.
    - U-Net employs a unique architecture and skip connections to maintain high-resolution details, performs well with fewer images, and ensures sharp and accurate segmentation.
    - The Dice score measures overlap between two samples, making it ideal for evaluating segmentation accuracy.
- Run `./model/train.py` to proceed with this step.


#### Step 6. Utilize Model
- This step utilizes the trained model to segment ships on the provided image.
- Run `./model/segmentation.py` to proceed with this step with default image.
- Use `--image_filepath <filepath>` flag in the terminal to pass your own image.


## Materials
- Kaggle Competition: https://www.kaggle.com/competitions/airbus-ship-detection
- TensorFlow Setup on Apple Silicon Mac: https://yashguptatech.medium.com/tensorflow-setup-on-apple-silicon-mac-m1-m1-pro-m1-max-661d4a6fbb77
