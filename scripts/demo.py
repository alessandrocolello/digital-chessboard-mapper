# Import libraries
import keras
from imageio.v3 import imread
from utils import resizeImage, applyPerspectiveTransformation, getGrid, stringToFEN
import numpy as np
import matplotlib.pyplot as plt
import chess
import chess.svg


# Define constants
NUM_KEYPOINTS = 4
SHAPE = (224, 224) # (height, width)

# Define demo image name and models paths
image_path = '../demo/demo_image.jpeg'
chessboard_highlighter_path = '../models/chessboard highlighter'
corner_detector_path = '../models/corner detector'
pieces_classifier_path = '../models/piece classifier'

# Define a dictionary that maps piece labels to class indices
classes = {'E': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}

# Create a list of piece labels
pieces = list(classes.keys())


# Loading models
print('Loading Chessboard Highlighter')
chessboard_highlighter = keras.models.load_model('../models/chessboard highlighter')
print('Loading Chessboard Highlighter')
corner_detector = keras.models.load_model('../models/corner detector')
print('Loading Pieces Classifier')
pieces_classifier = keras.models.load_model('../models/piece classifier')


# Read demo image
image = imread(image_path)

# Resize ROI
resized_image = resizeImage(image=image, height=SHAPE[0], width=SHAPE[1])

# Create chessboard mask
print('Highlighting chessboard')
highlighted_chessboard = chessboard_highlighter.predict(np.expand_dims(resized_image/255, axis=0), verbose=False)[0]

# Plot the image with highlighted chessboard
plt.imshow(highlighted_chessboard, cmap='gray')
# Turn off axis
plt.axis('off')
# Save the image
plt.savefig('../demo/highlighted_chessboard.png', bbox_inches='tight', pad_inches=0)

# Predict chessboard corners coordinates
print('Predicting corners')
pred = corner_detector.predict(np.expand_dims(highlighted_chessboard, axis=0), verbose=False)[0]
keypoints = np.float32([[pred[i]*image.shape[1], pred[i+1]*image.shape[0]] for i in range(0, 2*NUM_KEYPOINTS, 2)])

# Plot the image with highlighted chessboard
plt.figure()
plt.imshow(image, cmap='gray')
# Add the predicted corners
plt.scatter([pred[i]*image.shape[1] for i in range(0, 2*NUM_KEYPOINTS, 2)], [pred[i+1]*image.shape[0] for i in range(0, 2*NUM_KEYPOINTS, 2)])
# Turn off axis
plt.axis('off')
# Save the image
plt.savefig('../demo/predicted_corners.png', bbox_inches='tight', pad_inches=0)

# Apply perspective transformation to the ROI
print('Applying perspective transformation')
transformed_image = applyPerspectiveTransformation(input_image=image, input_keypoints=keypoints)

# Rotate the transformed ROI counterclockwise by 90 degrees
print('Rotating image')
rotated_transformed_image = np.rot90(transformed_image, 1)

# Plot the image with perspective transformed and rotated chessboard
plt.figure()
plt.imshow(rotated_transformed_image)
# Turn off axis
plt.axis('off')
# Save the image
plt.savefig('../demo/rotated_transformed_image.png', bbox_inches='tight', pad_inches=0)

# Extract square images from the rotated transformed ROI
print('Getting grid')
grid = getGrid(image=rotated_transformed_image, border=2)

# Create a figure with extracted squares images
fig = plt.figure(figsize=(100, 100))

for i in range(1, 65):

    fig.add_subplot(8, 8, i)
    plt.imshow(grid[i-1])
    plt.axis('off')

plt.savefig('../demo/extracted_squares.png', bbox_inches='tight', pad_inches=0)

position = ''

# Predict square images classes and update the current position
for i in range(64):

    print('Predicting square '+str(i))
    position += pieces[np.argmax(pieces_classifier.predict(np.expand_dims(grid[i]/255, axis=0), verbose=False)[0])]
    
    if (i+1)%8==0 and i<63:
        position += '/'

# Convert the string of the current position to FEN (Forsyth-Edwards Notation)
print('Converting string to FEN')
fen = stringToFEN(position)

# Create a chess board object from the FEN string
board = chess.Board(fen)

# Generate an SVG representation of the chessboard
svg_string = chess.svg.board(board=board, coordinates=False, colors={'square light': '#EAD7B7', 'square dark': '#AF8968'})

# Save the SVG representation of the chessboard
print('Saving chessboard')
with open('../demo/predicted_chessbord.svg', 'w') as f:

    f.write(svg_string)
