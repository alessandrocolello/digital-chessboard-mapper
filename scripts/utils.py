import tensorflow as tf
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import os
from imageio.v3 import imread, imwrite
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib import cm
import random


def resizeImageWithKeypoints(image, keypoints, height, width):
    """
    Resizes an image and rescales the coordinates of its keypoints.

    Args:
        image (numpy.array): A 3D numpy array representing an image.
        keypoints (imgaug.augmentables.kps.KeypointsOnImage): Container for all keypoints on an image.
        height (int): An integer indicating the height of the resized image.
        width (int): An integer indicating the width of the resized image.

    Returns:
        numpy.array: A 3D numpy array representing the resized image.
        numpy.array: A numpy array containing the rescaled keypoints.
    """
    
    # Create an image augmentation sequence that applies resizing to the input images
    seq = iaa.Sequential([iaa.Resize({'height': height, 'width': width})])

    # Return the transformed image and keypoints
    return seq(image=image, keypoints=keypoints)


def transformImageWithKeypoints(image, keypoints):
    """
    Applies a transformation to an image and its keypoints.

    Args:
        image (numpy.array): A 3D numpy array representing an image.
        keypoints (imgaug.augmentables.kps.KeypointsOnImage): Container for all keypoints on an image.

    Returns:
        numpy.array: A 3D numpy array representing the transformed image.
        numpy.array: A numpy array containing the transformed keypoints.
    """

    # Create an image augmentation sequence, consisting of a series of transformations
    seq = iaa.Sequential([

        iaa.OneOf([
            iaa.KeepSizeByResize(iaa.Affine(
                scale=(0.8, 1.2), # Scale the image by a factor between 0.8 and 1.2
                rotate=(-15, 15), # Rotate the image by an angle between -15 and 15 degrees
                translate_percent={'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, # Translate in both x and y directions
                mode='edge' # Handling of pixels outside the image boundary
            )), 
            iaa.geometric.PerspectiveTransform(scale=(0.1, 0.15), mode='replicate') # Apply perspective transformation
        ]),

        iaa.Cutout(nb_iterations=(1, 2), size=0.3, squared=False, fill_mode='gaussian'), # Remove rectangular portions from the image

        iaa.color.AddToBrightness((-30, 30)), # Adjust image brightness by adding a random value

        iaa.Sometimes(0.5, iaa.blur.GaussianBlur(sigma=(0, 0.5))) # Apply gaussian blur with a 50% probability
    ])

    # Apply the defined augmentation sequence to the input 'image' and 'keypoints'
    transformed_image, transformed_keypoints = seq(image=image, keypoints=keypoints)

    # Repeat the transformation until all transformed keypoints are within the image boundaries
    while not (all([tks[0]>0 for tks in transformed_keypoints.to_xy_array()]) & 
               all([tks[0]<image.shape[1] for tks in transformed_keypoints.to_xy_array()]) & 
               all([tks[1]>0 for tks in transformed_keypoints.to_xy_array()]) & 
               all([tks[1]<image.shape[0] for tks in transformed_keypoints.to_xy_array()])):
        transformed_image, transformed_keypoints = seq(image=image, keypoints=keypoints)

    # Return the transformed image and keypoints
    return transformed_image, transformed_keypoints


def createSet(folder, filenames, annotations, resizing_size, num_keypoints, num_augmentations):
    """
    Create a dataset by loading and processing images along with keypoints.

    Args:
        filenames (list): A list of image filenames to be processed.
        annotations (pandas.DataFrame): A DataFrame containing annotations for keypoints.

    Returns:
        tuple: A tuple containing three arrays:
            - images (numpy.array): An array of resized images.
            - keypoints (numpy.array): An array of keypoints corresponding to the images.
            - augmentation_flags (numpy.array): An array of boolean flags indicating if an image was augmented.
    """

    # Initialize empty lists to store images, keypoints, and augmentation flags
    images = []
    keypoints = []
    augmentation_flags = []

    # Iterate through each image filename
    for file_name in filenames:

        # Construct the full image path
        image_path = os.path.join(folder, file_name)

        # Read and convert the image to a numpy array
        image = imread(image_path)
        image = np.array(image)

        # Extract the coordinates of keypoints for the current image
        coordinates = annotations.loc[file_name]

        # Create KeypointsOnImage object from coordinates and image shape
        raw_keypoints = KeypointsOnImage([Keypoint(x=coordinates[i], y=coordinates[i+1]) for i in range(0, 2*num_keypoints, 2)], shape=image.shape)

        # Resize the image and keypoints to a specified size
        resized_image, resized_keypoints = resizeImageWithKeypoints(image=image, keypoints=raw_keypoints, height=resizing_size[1], width=resizing_size[0])

        # Append the resized image to the images list
        images.append(resized_image)

        # Convert the resized keypoints to a flat list and normalize
        resized_keypoints_list = resized_keypoints.to_xy_array().ravel().tolist()/np.array(num_keypoints*resizing_size[::-1])

        # Append the normalized keypoints to the keypoints list
        keypoints.append(resized_keypoints_list)

        # Append False to indicate that this image was not augmented
        augmentation_flags.append(False)

        # Iterate for a specified number of times
        for i in range(num_augmentations):

            # Apply data augmentation to the image and keypoints
            transformed_image, transformed_keypoints = transformImageWithKeypoints(image=resized_image, keypoints=resized_keypoints)
            
            # Append the augmented image to the images list
            images.append(transformed_image)

            # Convert the resized keypoints to a flat list and normalize
            transformed_keypoints_list = transformed_keypoints.to_xy_array().ravel().tolist()/np.array(num_keypoints*resizing_size[::-1])

            # Append the normalized keypoints to the keypoints list
            keypoints.append(transformed_keypoints_list)

            # Append True to indicate that this image was augmented
            augmentation_flags.append(True)

    # Convert lists to numpy arrays for images, keypoints, and augmentation flags
    images = np.array(images) 
    keypoints = np.array(keypoints)
    augmentation_flags = np.array(augmentation_flags)

    # Shuffle the dataset by randomizing the order of samples
    shuffled_indexes = np.random.permutation(range(images.shape[0]))

    # Return the shuffled images, keypoints, and augmentation flags
    return images[shuffled_indexes, :, :, :], keypoints[shuffled_indexes, :], augmentation_flags[shuffled_indexes]


def loadImagesWithKeypoints(dataset_folder, set_to_load, split=False):
    """
    Load images and their keypoints.

    Args:
        set_to_load (str): String that specifies which set to load: 'train_1', 'train_2', 'val' or 'test'.
        split (bool): Boolean value indicating wheter to split original images and their augmentations.

    Returns:
        2 or 4 numpy.array:
            If 'split' is True, returns a numpy.array containing the originals images, a numpy.array containing the original keypoints, 
            a numpy.array containing the augmented images, and a numpy.array containing the augmented keypoints.
            If 'split' is False, returns a numpy.array containing both the original images and the augmented images, a numpy.array containing
            both the original images and the augmented keypoints.
            In both cases, images are represented using RGB format and their values range from 0 to 1.
    """

    # Load image data from a npy file
    images = np.load(dataset_folder+set_to_load+'_images.npy')
    
    # Normalize images
    images = images/255

    # Load keypoints coordinates from a npy file
    keypoints = np.load(dataset_folder+set_to_load+'_keypoints.npy')

    # Load augmentation flags from a npy file
    augmented = np.load(dataset_folder+set_to_load+'_augmentation_flags.npy')

    # If 'split' is True, split the dataset into augmented and non-augmented subsets
    if split:

        return images[np.logical_not(augmented)], keypoints[np.logical_not(augmented)], images[augmented], keypoints[augmented]

    # If 'split' is False, return the entire dataset (images and keypoints)
    return images, keypoints


def generateChessboardMask(keypoints, height, width):
    """
    Generate a black and white chessboard mask image based on keypoints.

    Args:
        keypoints (numpy.array): A 2D numpy array containing the coordinates of the four corners of the chessboard expressed as percentages of the total image width and height.
        height (int): The height of the output image in pixels.
        width (int): The width of the output image in pixels.

    Returns:
        numpy.array: A numpy array representing a black and white chessboard mask image.
    """

    # Create an empty black image with the specified dimensions (height and width) where each pixel has an initial value of 0 (black)
    Y = np.zeros((keypoints.shape[0], height, width), dtype=np.uint8)

    # Iterate over each set of keypoints
    for i in range(keypoints.shape[0]):

        # Extract the corners of the chessboard by converting the percentage-based coordinates to pixel coordinates, rounding them to the nearest integer
        corners = np.array([(round(keypoints[i, j]*width), round(keypoints[i, j+1]*height)) for j in range(0, 8, 2)])

        # Calculate the convex hull to define the chessboard outline
        hull = cv2.convexHull(corners)

        # Draw the convex hull on the empty black image, filling it with white color (255, 255, 255)
        cv2.drawContours(Y[i], [hull], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Return the normalized images
    return Y/255


def loadChessPieceImages(dataset_folder, set_to_load, split=False):
    """
    Load images of chess pieces.

    Args:
        set_to_load (str): String that specifies which set to load: 'train', 'val' or 'test'.
        split (bool): Boolean value indicating wheter to split original images and their augmentations.

    Returns:
        2 or 4 numpy.array:
            If 'split' is True, returns a numpy.array containing the originals images, a numpy.array containing the original keypoints, 
            a numpy.array containing the augmented images, and a numpy.array containing the augmented keypoints.
            If 'split' is False, returns a numpy.array containing both the original images and the augmented images, a numpy.array containing
            both the original images and the augmented keypoints.
            In both cases, images are represented using RGB format and their values range from 0 to 1.
    """

    # Load image data from a npy file
    images = np.load(dataset_folder+set_to_load+'_images.npy')

    # Normalize images 
    images = images/255

    # Load labels from a npy file
    labels = np.load(dataset_folder+set_to_load+'_labels.npy')
    
    # Convert labels to one-hot encoded format
    labels = to_categorical(labels, num_classes=len(np.unique(labels)))

    # Load augmentation flags froma npy file
    augmented = np.load(dataset_folder+set_to_load+'_augmentation_flags.npy')

    # If 'split' is True, split the dataset into augmented and non-augmented subsets
    if split:
        
        return images[np.logical_not(augmented)], labels[np.logical_not(augmented)], images[augmented], labels[augmented]

    # If 'split' is False, return the entire dataset (images and labels)
    return images, labels


def saveImage(image, image_path):
    """
    Saves an image.

    Args:
        image (numpy.array): A numpy array representing an image.
        image_path (str): Path of the image to save.

    Returns:
        None: This function does not return any value.
    """

    # Extract the image folder path from the image_path
    image_folder = '/'.join(image_path.split('/')[:-1])

    # Create the image folder if it does not exist
    if not os.path.exists(image_folder):
       os.makedirs(image_folder)

    # Write the image
    imwrite(image_path, image)


def resizeImage(image, height, width):
    """
    Resizes an image.

    Args:
        image (numpy.array): A 3D numpy array representing an image.
        height (int): Integer indicating the height of the resized image.
        width (int): Integer indicating the width of the resized image.

    Returns:
        numpy.array: A 3D numpy array representing the resized image.
    """

    seq = iaa.Sequential([iaa.Resize({'height': height, 'width': width})])

    return seq(image=image)


def transformImage(image):
    """
    Applies a transformation to an image.

    Args:
        image (numpy.array): A 3D numpy array representing an image.

    Returns:
        numpy.array: A 3D numpy array representing the transformed image.
    """

    seq = iaa.Sequential([

        iaa.KeepSizeByResize(iaa.Affine(scale=(0.8, 1.1), rotate=(-5, 5), translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)}, mode='edge')),

        iaa.AddToBrightness((-30, 30)),

        iaa.Cutout(nb_iterations=(0, 3), size=0.2, squared=False, fill_mode='gaussian'), 

        iaa.Fliplr(p=0.5),
        
        iaa.Flipud(p=0.5)
    ])

    return seq(image=image)


def applyPerspectiveTransformation(input_image, input_keypoints):
    """
    Apply perspective transformation to an input image based on given keypoints.

    Args:
        input_image (numpy.array): A 3D numpy array representing the input image.
        input_keypoints (numpy.array): A 2D numpy array containing input keypoints that define the transformation.

    Returns:
        numpy.array: A 3D numpy array representing the transformed image.
    """

    # Define the size of the output image (both width and height)
    # l = 464
    l = 554

    # Define output keypoints representing the four corners of a square
    output_keypoints = np.float32([[0, 0], [l, 0], [0, l], [l, l]])
    
    # Calculate the perspective transformation matrix 'M' using input and output keypoints
    M = cv2.getPerspectiveTransform(input_keypoints, output_keypoints)
    
    # Apply the perspective transformation to the input image
    output = cv2.warpPerspective(input_image, M, (l, l))

    # Return the transformed image
    return output


def getGrid(image, border):
    """
    Split an image of a chessboard in a 8x8 grid. Each cell is supposed to contain a square of the board.

    Args:
        image (numpy.array): A 3D numpy array representing an image.
        border (int): Integer indicating the border.

    Returns:
        list: List of 64 numpy.arrays.
    """

    # Get the height and width of the input image    
    height, width, _ = image.shape

    # Define the horizontal (H) and vertical (W) splitting ranges for the image. 'border' is used to add some padding to each subregion.
    # H = [[i.min()-border, i.max()+border] for i in np.array_split(np.arange(28, height-28), 8)]
    # W = [[i.min()-border, i.max()+border] for i in np.array_split(np.arange(32, width-24), 8)]

    H = [[i.min()-border, i.max()+border] for i in np.array_split(np.arange(33, height-33), 8)]
    W = [[i.min()-border, i.max()+border] for i in np.array_split(np.arange(38, width-28), 8)]

    # Create a grid of subimages by cropping the input image based on the defined ranges.
    grid = [image[i:ii, j:jj, :] for i, ii in H for j, jj in W]

    # Return the grid, which contains a list of subimages.
    return grid


def stringToFEN(string):
    """
    Adjusts a string to comply with FEN notation.

    Args:
        string (str): String indicating which piece is on each square.

    Returns:
        str: String describing the position using the FEN notation.

    Example:
        >>> string = 'EEEnEEkN/EEEEpEEn/EEEPrEEE/bNEEEEEE/qEREEEEE/EEEBEEBb/PRQEEEEr/EpEEEEK'
        >>> stringToFEN(string)
        '3n2kN/4p2n/3Pr3/bN6/q1R5/3B2Bb/PRQ4r/1p5K'
    """

    for i in range(8, 0, -1): 

        string = string.replace(i*'E', str(i))

    # Return the modified 'string' after replacing the longest consecutive 'e' sequence with its length
    return string


def visualizeIntermediateActivations(model, image, output_folder, layer_name, images_per_row=16, margin=1):
    """
    Generate intermediate activations of convolutional layers and save them as an image.

    Args:
        model (keras.Model): The Convolutional Neural Network model.
        image (numpy.array): A 3D numpy array containing an image in RGB format.
        output_folder (string): Path to the folder where to save the output image.
        layer_name (string): Convolutional layer name whose activations are to be displayed.
        images_per_row (int): Number of convolutional layer activations to be placed in each row.
        margin (int): Number of pixels separating displays of activations.

    Returns:
        None: This function saves the visualizations as an image and does not return any value.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the Viridis color map for visualization
    viridis_cmap = cm.get_cmap('viridis')

    # Create a model that will output the activations of the selected layer
    visualization_model = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Generate the activations for the input image
    activations = visualization_model.predict(tf.expand_dims(image, axis=0), verbose=False)
        
    # Get the dimensions of the activations tensor
    _, activations_height, activations_width, n_filters = activations.shape
        
    # Calculate the number of rows and columns to display the filters
    n_rows = n_filters//images_per_row + n_filters%images_per_row
    n_cols = min(images_per_row, n_filters)

    # Create an image to store the visualizations
    images = np.full((n_rows*activations_height+(n_rows-1)*margin, n_cols*activations_width+(n_cols-1)*margin, 3), 255)

    for i in range(n_rows):

        for j in range(n_cols):

            if i*images_per_row+j < n_filters:

                activation = activations[0, :, :, i*images_per_row+j]

                # Normalize the layer activation
                activation = (activation-np.min(activation))/(np.max(activation)-np.min(activation)+1e-5)

                activation = viridis_cmap(activation)[..., :3]

                activation = (activation*255).astype('uint8')

                images[i*(activations_height+margin):i*(activations_height+margin)+activations_height, j*(activations_width+margin):j*(activations_width+margin)+activations_width, :] = activation

    # Save the intermediate activations as an image
    tf.keras.preprocessing.image.save_img(output_folder+layer_name+'.png', images, scale=False)


def visualizeFiltersResponses(model, output_folder, layer_name, images_per_row, input_shape, margin=1):
    """
    Generate and save visual patterns that CNN filters respond to.

    Args:
        model (keras.Model): The Convolutional Neural Network model.
        output_folder (str): Path to the folder where visualizations will be saved.
        layer_name (str): Convolutional layer name for which filter responses will be visualized.
        images_per_row (int): Number of filter responses to be displayed in each row.
        input_shape (tuple): Shape of the input images (height, width, channels).
        margin (int): Number of pixels separating displayed filter responses.

    Returns:
        None: This function saves the visualizations as an image and does not return any value.
    """

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Constants for gradient ascent optimization
    ITERATIONS = 30
    LEARNING_RATE = 10

    # Get the selected layer from the model
    layer = model.get_layer(name=layer_name)

    # Get the dimensions of the filter responses
    _, activations_height, activations_width, n_filters = layer.output.shape

    # Calculate the number of rows and columns for filter response display 
    n_rows = n_filters//images_per_row+int(n_filters<images_per_row)
    n_cols = min(images_per_row, n_filters)

    # List to store filter response images
    images = []

    for filter_index in range(n_filters):

        # Initialize a gray image with random noise for optimization
        input_image = tf.random.uniform((1, input_shape[0], input_shape[1], input_shape[2]))

        print('Processing filter '+str(filter_index)+' of layer '+layer_name)
        for iteration in range(ITERATIONS):
            
            # Compute gradient ascent to maximize filter activation
            with tf.GradientTape() as tape:
                
                tape.watch(input_image)
                
                feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)
                
                activation = feature_extractor(input_image)
                
                loss = tf.reduce_mean(activation[:, :, :, filter_index])
            
            # Compute gradients
            grads = tape.gradient(loss, input_image)
            
            # Normalize gradients
            grads = tf.math.l2_normalize(grads)
            
            # Update the image
            input_image += LEARNING_RATE*grads
        
        # Deprocess the image (scale and clip)
        input_image = 0.15*(input_image[0].numpy()-input_image[0].numpy().mean())/(input_image[0].numpy().std()+1e-5)
        input_image += 0.5
        input_image = np.clip(input_image, 0, 1)

        # Convert the image to RGB format
        input_image *= 255
        input_image = np.clip(input_image, 0, 255).astype('uint8')
        
        images.append(input_image)

    # Create an image to display the filter responses
    filter_images = np.full((n_rows*input_shape[0]+(n_rows-1)*margin, n_cols*input_shape[1]+(n_cols-1)*margin, 3), 255)
    
    for i in range(n_rows):
        for j in range(n_cols):

            if i*n_cols+j < n_filters:
                filter_images[i*(input_shape[0]+margin):i*(input_shape[0]+margin)+input_shape[0], j*(input_shape[1]+margin):j*(input_shape[1]+margin)+input_shape[1], :] = images[i*n_cols+j]        

    # Save the filter response visualizations as an image
    tf.keras.preprocessing.image.save_img(output_folder+layer_name+'.png', filter_images)


def visualizeHeatmap(model, last_conv_layer_name, image, pred_index=None, output_path='heatmap.png', alpha=0.1):

    """
    Generate and save a class activation heatmap overlaid on an input image.

    Args:
        model (keras.Model): The Convolutional Neural Network model.
        last_conv_layer_name (str): The name of the last convolutional layer.
        image (numpy.array): The input image as a 3D numpy array.
        pred_index (int): The predicted class index (optional).
        output_path (str): Path to save the output heatmap image.
        alpha (float): Transparency factor for overlaying the heatmap on the image.

    Returns:
        None: This function saves the heatmap image and does not return any value.
    """

    output_folder = '/'.join(output_path.split('/')[:-1])

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Expand dimensions to make a batch of one image
    image = np.expand_dims(image, axis = 0)

    # Create a model that extracts the activations of the last convolutional layer and the output predictions
    grad_model = keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Calculate gradients of the top predicted class (or chosen) with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate the gradient of the output neuron with respect to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Compute the mean intensity of the gradients over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by its importance to the top predicted class
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]

    # Sum all the channels to obtain the class activation heatmap
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1 for visualization
    heatmap = tf.maximum(heatmap, 0)/tf.math.reduce_max(heatmap)

    # Rescale the heatmap to a range of 0-255
    heatmap = np.uint8(255*heatmap) 

    # Create a jet colormap for colorizing the heatmap
    jet = plt.colormaps.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image[0, :, :, :].shape[1], image[0, :, :, :].shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap) 

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap*alpha+image[0,:,:,:]
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(output_path)






