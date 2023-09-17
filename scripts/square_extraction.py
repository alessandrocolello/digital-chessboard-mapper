# Import libraries
from imageio.v2 import imread
import glob
import numpy as np
import keras
import pandas as pd
from utils import resizeImage, saveImage, applyPerspectiveTransformation, getGrid
import matplotlib.pyplot as plt

# Define contants
NUM_KEYPOINTS = 4

# Define paths to folders
pieces_images_folder = '../files/piece classification/positions/'
raw_pieces_images_folder = '../files/piece classification/raw squares images/'


# Load models
chessboard_highlighter = keras.models.load_model('../models/chessboard highlighter')
corner_detector = keras.models.load_model('../models/corner detector')

# Define a dictionary that maps FEN chess piece symbols to their corresponding names
pieces = {'e': 'Empty Square', 
          'P': 'White Pawn', 'N': 'White Knight', 'B': 'White Bishop', 'R': 'White Rook', 'Q': 'White Queen', 'K': 'White King', 
          'p': 'Black Pawn', 'n': 'Black Knight', 'b': 'Black Bishop', 'r': 'Black Rook', 'q': 'Black Queen', 'k': 'Black King'}

# Load positions data from a CSV file containing FEN positions
positions = pd.read_csv(pieces_images_folder+'FENs.csv', header=None, index_col=0)

# Iterate over image files in the specified folder
for image_path in glob.glob(pieces_images_folder+'*.jpeg'):

    # Extract the image name from the path
    image_name = image_path.split('/')[-1]

    # Read the image
    image = imread(image_path)

    # Convert the image to a numpy array
    image = np.asarray(image)

    # Resize the image to a fixed height and width
    resized_image = resizeImage(image=image, height=224, width=224)

    # Predict the highlighted chessboard
    highlighted_chessboard = chessboard_highlighter.predict(np.expand_dims(resized_image, 0)/255, verbose=False)[0]

    # Predict the keypoints (corners) on the chessboard
    prediction = corner_detector.predict(np.expand_dims(highlighted_chessboard, 0), verbose=False)[0]

    # Convert the predicted keypoints to pixel coordinates based on the original image's shape
    predicted_keypoints = np.float32([[prediction[j]*image.shape[1], prediction[j+1]*image.shape[0]] for j in range(0, 2*NUM_KEYPOINTS, 2)])

    # Apply perspective transformation to correct the chessboard's perspective.
    transformed_image = applyPerspectiveTransformation(input_image=image, input_keypoints=predicted_keypoints)

    # Rotate the transformed image 90 degrees clockwise
    rotated_transformed_image = np.rot90(transformed_image)

    # Get the FEN string for the current image
    fen = positions.loc[image_name.split('.')[0]][1]

    # Create a simplified FEN string where digits represent empty squares
    string = "".join([int(ch)*'e' if ch.isdigit() else ch for ch in fen]).replace('/', '')

    # Split the rotated image into a grid of 64 squares
    grid = getGrid(image=rotated_transformed_image, border=2)

    # Iterate over the squares in the grid
    for j in range(64):

        # If the square is not empty, save the square as an image in the corresponding piece's folder
        if string[j]!='e': 
            saveImage(image=grid[j], image_path=raw_pieces_images_folder+pieces[string[j]]+'/'+str(j)+'_'+image_name)

        # If the square is empty and a random condition is met, save the square as an image in the 'Empty Square' folder
        if string[j]=='e' and np.random.binomial(1, 1/string.count('e')):         
           saveImage(image=grid[j], image_path=raw_pieces_images_folder+'Empty Square/'+str(j)+'_'+image_name)

