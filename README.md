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
   ![Real chessboard image](demo/demo_image.jpeg)

2. Mask of the generated chessboard.
   ![Highlighted chessboard](demo/chessboard_mask.png)

3. Predicted corners of the chessboard.
   ![Detected cheessboard corners](demo/predicted_corners.png)

4. Perspective transformed and rotated image.
   ![Perspective transformed and rotated image](demo/rotated_transformed_image.png)
   
5. Extracted squares.
   ![Extracted squares](demo/extracted_squares.png)

6. Predicted chessboard.
   ![Predicted digital chessboard](demo/predicted_chessboard.png)

[![Real-Time Video Application Showcase](https://img.youtube.com/vi/ZVWDbiI0f2M/0.jpg)](https://www.youtube.com/watch?v=ZVWDbiI0f2M)
