# Import libraries
import pandas as pd
import numpy as np
from utils import createSet
import os
import random

# Define constants
RESIZING_SIZE = [224, 224] # [height, width]
NUM_AUGMENTATIONS = 10
NUM_KEYPOINTS = 4

# Set numpy seed
np.random.seed(42)

# Define paths to folders and files.
raw_images_folder = '../files/chessboard detection/raw images/'
dataset_folder = '../files/chessboard detection/dataset/'
raw_annotations_path = '../files/chessboard detection/annotations.csv'


# Check if the dataset folder exists, and create it if not
if not os.path.exists(dataset_folder):
   os.makedirs(dataset_folder)

# Define the ratios for splitting the dataset into training, validation, and test sets
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# Read and process annotations from the CSV file
annotations = pd.read_csv(raw_annotations_path, header=None)
annotations.columns = ['corner', 'x', 'y', 'filename', 'width', 'height']
annotations = annotations.pivot(index='filename', columns='corner', values=['x', 'y'])
annotations = annotations[[('x','A1'), ('y','A1'),
                           ('x','A8'), ('y','A8'),
                           ('x','H1'), ('y','H1'),
                           ('x','H8'), ('y','H8')]]

# Get the total number of images
num_images = annotations.shape[0]

# List image files in the raw images folder
image_files = [d for d in os.listdir(raw_images_folder) if not d.startswith('.') and d.endswith('.jpeg')]
num_images = len(image_files)

# Shuffle the list of image files randomly
random.shuffle(image_files)

# Calculate the number of images for the training and validation sets
train_count = int(train_ratio*num_images)
val_count = int(val_ratio*num_images)

# Split the image files into training, validation, and test sets
train_files = image_files[:train_count]
val_files = image_files[train_count:(train_count+val_count)]
test_files = image_files[(train_count+val_count):]


# Create the training set
print('Creating the training set')
train_images, train_keypoints, train_augmentation_flags = createSet(folder=raw_images_folder, filenames=train_files, annotations=annotations,
                                                                    resizing_size=RESIZING_SIZE, num_keypoints=NUM_KEYPOINTS, num_augmentations=NUM_AUGMENTATIONS)
np.save(os.path.join(dataset_folder, 'train_images.npy'), train_images)
np.save(os.path.join(dataset_folder, 'train_keypoints.npy'), train_keypoints)
np.save(os.path.join(dataset_folder, 'train_augmentation_flags.npy'), train_augmentation_flags)


# Create the validation set
print('Creating the validation set')
val_images, val_keypoints, val_augmentation_flags = createSet(folder=raw_images_folder, filenames=val_files, annotations=annotations, 
                                                              resizing_size=RESIZING_SIZE, num_keypoints=NUM_KEYPOINTS, num_augmentations=NUM_AUGMENTATIONS)
np.save(os.path.join(dataset_folder, 'val_images.npy'), val_images)
np.save(os.path.join(dataset_folder, 'val_keypoints.npy'), val_keypoints)
np.save(os.path.join(dataset_folder, 'val_augmentation_flags.npy'), val_augmentation_flags)


# Create the test set
print('Creating the test set')
test_images, test_keypoints, test_augmentation_flags = createSet(folder=raw_images_folder, filenames=test_files, annotations=annotations, 
                                                                 resizing_size=RESIZING_SIZE, num_keypoints=NUM_KEYPOINTS, num_augmentations=NUM_AUGMENTATIONS)
np.save(os.path.join(dataset_folder, 'test_images.npy'), test_images)
np.save(os.path.join(dataset_folder, 'test_keypoints.npy'), test_keypoints)
np.save(os.path.join(dataset_folder, 'test_augmentation_flags.npy'), test_augmentation_flags)
