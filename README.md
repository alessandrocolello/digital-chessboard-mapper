# Digital Chessboard Mapper

During many chess tournaments, expensive electronic chessboards are used to broadcast and analyse live games. This project uses machine learning to avoid the acquisition
of such chessboards. The devised programme uses three convolutional neural networks (CNN) to recognise a real chessboard from a video transmission and highlight it (Chessboard Highlighter), to locate the four corners of the chessboard, so that the 64 squares can be precisely extracted (Corner Detector) and, finally, to recognise the piece on each square (Piece Classifier). This determines the current position of the chessboard in Forsyth-Edwards Notation (FEN). With this notation, it is possible to visualise the position within a virtual chessboard.

## Table of Contents

- [Features](#features)
- [Screenshots/Demo](#screenshots-demo)

## Features

- Chessboard highlighter: Convolutional Neural Network (CNN) that generates a chessboard mask from a real chessboard image;
- Corner detector: CNN that extracts the coordinates of the four chessboard corners from a chessboard mask;
- Piece classifier: CNN that classifies square images into the following categories: empty square, white pawn, white knight, white bishop, white rook, white queen, white king, black pawn, black knight, black bishop, black rook, black queen, and black king.

## Screenshots/Demo

1. Image of a chessboard taken as input.
   <center><img src="demo/demo_image.jpeg" width="300" height="300"></center>

3. Mask of the generated chessboard.
   <center><img src="demo/chessboard_mask.png" width="300" height="300"></center>

5. Predicted corners of the chessboard.
   <center><img src="demo/predicted_corners.png" width="300" height="300"></center>

7. Perspective transformed and rotated image.
   <center><img src="demo/rotated_transformed_image.png" width="300" height="300"></center>
   
9. Extracted squares.
    <center><img src="demo/extracteed_squares.png" width="300" height="300"></center>

11. Predicted chessboard.
    <center><img src="demo/predicted_chessboard.png" width="300" height="300"></center>

### Real-Time Video Application Showcase:
[![Watch the video](https://img.youtube.com/vi/ZVWDbiI0f2M/default.jpg)](https://youtu.be/ZVWDbiI0f2M)
