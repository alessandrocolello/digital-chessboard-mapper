# Import libraries
import keras
import numpy as np
import chess
import chess.svg
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from PIL import ImageTk
import cairosvg
from utils import resizeImage, applyPerspectiveTransformation, getGrid, stringToFEN
import io


# Define constants
NUM_KEYPOINTS = 4
SHAPE = (224, 224) # (height, width)
WINDOW_WIDTH = 400 # width of the tkinter window
WINDOW_HEIGHT = 768 # height of the tkinter window
X_POSITION = 768 # x position of the tkinter window
CANVAS_SIDE = 380 # length of the side of a canvas

# Define a dictionary that maps piece labels to class indices
classes = {'E': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, 'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}

# Create a list of piece labels
pieces = list(classes.keys())

# Define video capture settings
username = 'admin'
password = 'admin'
ip_address = '192.168.1.27'
port = '8081'


# Loading models
print('Loading Chessboard Highlighter')
chessboard_highlighter = keras.models.load_model('../models/chessboard highlighter')
print('Loading Chessboard Highlighter')
corner_detector = keras.models.load_model('../models/corner detector')
print('Loading Pieces Classifier')
pieces_classifier = keras.models.load_model('../models/piece classifier')


# Initialize video capture from a URL
cap = cv2.VideoCapture('http://'+username+':'+password+'@'+ip_address+':'+port)

# Create a Tkinter window
root = tk.Tk()
root.title('')
root.geometry(str(WINDOW_WIDTH)+'x'+str(WINDOW_HEIGHT)+'+'+str(X_POSITION)+'+0')

# Create the first canvas and place it at the top
canvas1 = tk.Canvas(root, width=CANVAS_SIDE, height=CANVAS_SIDE)
canvas1.pack(side=tk.TOP)

# Create the second canvas and place it at the bottom
canvas2 = tk.Canvas(root, width=CANVAS_SIDE, height=CANVAS_SIDE)
canvas2.pack(side=tk.BOTTOM)


while True:

    # Read a frame from the video capture
    ret, frame = cap.read()

    # Break the loop if no frame is captured
    if not ret:
        break

    # Get the original video's width and height
    height, width, _ = frame.shape

    # Calculate the cropping parameters
    crop_1 = abs(height-width)//2
    crop_2 = crop_1+min(height, width)

    # Crop the frame
    roi = frame[crop_1:crop_2, :]

    # Display the resized frame
    cv2.imshow('Video Stream', roi)

    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed
    if key == ord('c'):

        # Convert from BGR ro RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Resize ROI
        resized_rgb_roi = resizeImage(image=rgb_roi, height=SHAPE[0], width=SHAPE[1])

        # Create chessboard mask
        print('Highlighting chessboard')
        highlighted_chessboard = chessboard_highlighter.predict(np.expand_dims(resized_rgb_roi/255, axis=0), verbose=False)[0]

        # Predict chessboard corners coordinates
        print('Predicting corners')
        pred = corner_detector.predict(np.expand_dims(highlighted_chessboard, axis=0), verbose=False)[0]
        keypoints = np.float32([[pred[i]*rgb_roi.shape[1], pred[i+1]*rgb_roi.shape[0]] for i in range(0, 2*NUM_KEYPOINTS, 2)])

        # Apply perspective transformation to the ROI
        print('Applying perspective transformation')
        transformed_rgb_roi = applyPerspectiveTransformation(input_image=rgb_roi, input_keypoints=keypoints)

        # Rotate the transformed ROI counterclockwise by 90 degrees
        print('Rotating image')
        rotated_transformed_rgb_roi = np.rot90(transformed_rgb_roi, 1)

        # Convert the rotated transformed ROI to a PIL image and resize it to match canvas1 size
        pil_image = Image.fromarray(rotated_transformed_rgb_roi).resize((CANVAS_SIDE, CANVAS_SIDE), Image.LANCZOS)

        # Convert the PIL image to a PhotoImage object
        photo1 = ImageTk.PhotoImage(pil_image)

        # Display the image on canvas1
        canvas1.create_image(0, 0, anchor=tk.NW, image=photo1)
        
        # Update canvas1
        canvas1.update()


    # If the space bar is pressed (key code for space is 32)
    if key == 32:

        # Convert from BGR ro RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Apply perspective transformation to the ROI
        print('Applying perspective transformation')
        transformed_rgb_roi = applyPerspectiveTransformation(input_image=rgb_roi, input_keypoints=keypoints)

        # Rotate the transformed ROI counterclockwise by 90 degrees
        print('Rotating image')
        rotated_transformed_rgb_roi = np.rot90(transformed_rgb_roi, 1)

        # Extract square images from the rotated transformed ROI
        print('Getting grid')
        grid = getGrid(image=rotated_transformed_rgb_roi, border=2)

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
        
        # Convert the SVG string to a PNG image using cairosvg
        png_image = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))

        # Load the PNG image using PIL and resize it to match canvas2 size
        pil_image = Image.open(io.BytesIO(png_image)).resize((CANVAS_SIDE, CANVAS_SIDE), Image.LANCZOS)

        # Convert the PNG image to a PhotoImage object for Tkinter
        photo2 = ImageTk.PhotoImage(pil_image)

        # Display the image on canvas2
        canvas2.create_image(0, 0, anchor=tk.NW, image=photo2)

        # Update canvas2
        canvas2.update()

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
