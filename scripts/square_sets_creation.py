# Import libraries
import os
from imageio.v2 import imread
import numpy as np
from utils import transformImage
import random


# Define paths to folders for raw and processed images
raw_images_folder = '../files/piece classification/raw squares images/'
dataset_folder = '../files/piece classification/dataset/'

# Create the dataset folder if it doesn't exist
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)


# Define the number of augmentations for each class
NUM_AUGMENTATIONS = {'Empty Square': 1, 
                     'White Pawn': 3, 'White Knight': 12, 'White Bishop': 12, 'White Rook': 12, 'White Queen': 12, 'White King': 12, 
                     'Black Pawn': 3, 'Black Knight': 12, 'Black Bishop': 12, 'Black Rook': 12, 'Black Queen': 12, 'Black King': 12} 


# Define a mapping of class names to class labels
classes = {'Empty Square': 0, 
           'White Pawn': 1, 'White Knight': 2, 'White Bishop': 3, 'White Rook': 4, 'White Queen': 5, 'White King': 6, 
           'Black Pawn': 7, 'Black Knight': 8, 'Black Bishop': 9, 'Black Rook': 10, 'Black Queen': 11, 'Black King': 12} 

# Define the ratios for splitting the dataset into training, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Initialize lists to store images, labels, and augmentation flags for training, validation, and test sets
train_images = []
train_labels = []
train_augmentation_flags = []

val_images = []
val_labels = []
val_augmentation_flags = []

test_images = []
test_labels = []
test_augmentation_flags = []


# Iterate over class folders containing chess piece images
for class_name in [d for d in os.listdir(raw_images_folder) if not d.startswith('.')]:

    # Define the path to the class folder and list image files
    class_folder = os.path.join(raw_images_folder, class_name)
    image_files = [d for d in os.listdir(class_folder) if not d.startswith('.') and d.endswith('.jpeg')]
    
    # Get the number of images
    num_images = len(image_files)
    
    # Calculate the number of images for the training and validation sets
    train_count = int(train_ratio*num_images)
    val_count = int(val_ratio*num_images)
    
    # Split the image files into training, validation, and test sets
    train_files = image_files[:train_count]
    val_files = image_files[train_count:(train_count+val_count)]
    test_files = image_files[(train_count+val_count):]
    
    # Iterate over images in the training set
    for file_name in train_files:
        image_path = os.path.join(class_folder, file_name)
        image = imread(image_path)
        image = np.array(image)
        train_images.append(image)
        train_labels.append(classes[class_name])
        train_augmentation_flags.append(False)

        # Apply data augmentation for the specified number of times
        for i in range(NUM_AUGMENTATIONS[class_name]):
            transformed_image = transformImage(image=image)
            train_images.append(transformed_image)
            train_labels.append(classes[class_name])
            train_augmentation_flags.append(True)

    # Iterate over images in the validation set
    for file_name in val_files:
        image_path = os.path.join(class_folder, file_name)
        image = imread(image_path)
        image = np.array(image)
        val_images.append(image)
        val_labels.append(classes[class_name])
        val_augmentation_flags.append(False)

        # Apply data augmentation for the specified number of times
        for i in range(NUM_AUGMENTATIONS[class_name]):
            transformed_image = transformImage(image=image)
            val_images.append(transformed_image)
            val_labels.append(classes[class_name])
            val_augmentation_flags.append(True)
    
    # Iterate over images in the test set
    for file_name in test_files:
        image_path = os.path.join(class_folder, file_name)
        image = imread(image_path)
        image = np.array(image)
        test_images.append(image)
        test_labels.append(classes[class_name])
        test_augmentation_flags.append(False)

        # Apply data augmentation for the specified number of times
        for i in range(NUM_AUGMENTATIONS[class_name]):
            transformed_image = transformImage(image=image)
            test_images.append(transformed_image)
            test_labels.append(classes[class_name])
            test_augmentation_flags.append(True)


# Convert the lists to numpy arrays for images, labels, and augmentation flags
train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_augmentation_flags = np.array(train_augmentation_flags)

# Shuffle the training set by randomizing the order of samples
shuffled_indexes = np.random.permutation(range(train_images.shape[0]))
train_images = train_images[shuffled_indexes, :, :, :]
train_labels = train_labels[shuffled_indexes]
train_augmentation_flags = train_augmentation_flags[shuffled_indexes]

# Save the training set as numpy arrays
np.save(os.path.join(dataset_folder, 'train_images.npy'), train_images)
np.save(os.path.join(dataset_folder, 'train_labels.npy'), train_labels)
np.save(os.path.join(dataset_folder, 'train_augmentation_flags.npy'), train_augmentation_flags)


# Convert the lists to numpy arrays for images, labels, and augmentation flags
val_images = np.array(val_images)
val_labels = np.array(val_labels)
val_augmentation_flags = np.array(val_augmentation_flags)

# Shuffle the validation set by randomizing the order of samples
shuffled_indexes = np.random.permutation(range(val_images.shape[0]))
val_images = val_images[shuffled_indexes, :, :, :]
val_labels = val_labels[shuffled_indexes]
val_augmentation_flags = val_augmentation_flags[shuffled_indexes]

# Save the validation set as numpy arrays
np.save(os.path.join(dataset_folder, 'val_images.npy'), val_images)
np.save(os.path.join(dataset_folder, 'val_labels.npy'), val_labels)
np.save(os.path.join(dataset_folder, 'val_augmentation_flags.npy'), val_augmentation_flags)


# Convert the lists to numpy arrays for images, labels, and augmentation flags
test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_augmentation_flags = np.array(test_augmentation_flags)

# Shuffle the test set by randomizing the order of samples
shuffled_indexes = np.random.permutation(range(test_images.shape[0]))
test_images = test_images[shuffled_indexes, :, :, :]
test_labels = test_labels[shuffled_indexes]
test_augmentation_flags = test_augmentation_flags[shuffled_indexes]

# Save the test set as numpy arrays
np.save(os.path.join(dataset_folder, 'test_images.npy'), test_images)
np.save(os.path.join(dataset_folder, 'test_labels.npy'), test_labels)
np.save(os.path.join(dataset_folder, 'test_augmentation_flags.npy'), test_augmentation_flags)
