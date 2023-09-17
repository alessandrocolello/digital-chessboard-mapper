# Importing used libraries
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from utils import loadImagesWithKeypoints, generateChessboardMask, visualizeIntermediateActivations, visualizeFiltersResponses


# Clearing session and setting random seed
keras.backend.clear_session()
keras.utils.set_random_seed(42)


# Defining constants
SHAPE = [224, 224, 3] # [height, width, channels]
BATCH_SIZE = 16
EPOCHS = 20
dataset_folder = '../files/chessboard detection/dataset/'


# Loading raw training, validation, and test sets
X_train, y_train = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='train')
X_val_no_aug, y_val_no_aug, X_val_aug, y_val_aug = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='val', split=True)
X_test_no_aug, y_test_no_aug, X_test_aug, y_test_aug = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='test', split=True)


# Generating chessboard masks from raw training, validation, and test sets keypoints coordinates
Y_train = generateChessboardMask(keypoints=y_train, height=SHAPE[0], width=SHAPE[1])
Y_val_no_aug = generateChessboardMask(keypoints=y_val_no_aug, height=SHAPE[0], width=SHAPE[1])
Y_val_aug = generateChessboardMask(keypoints=y_val_aug, height=SHAPE[0], width=SHAPE[1])
Y_test_no_aug = generateChessboardMask(keypoints=y_test_no_aug, height=SHAPE[0], width=SHAPE[1])
Y_test_aug = generateChessboardMask(keypoints=y_test_aug, height=SHAPE[0], width=SHAPE[1])


# Displaying two raw training images and their respective target images
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X_train[0])
axes[0, 0].set_title('X_train[0]')

axes[0, 1].imshow(Y_train[0], cmap='gray')
axes[0, 1].set_title('Y_train[0]')

axes[1, 0].imshow(X_train[1])
axes[1, 0].set_title('X_train[1]')

axes[1, 1].imshow(Y_train[1], cmap='gray')
axes[1, 1].set_title('Y_train[1]')

plt.tight_layout()
plt.show()


# Create a Sequential model with the name 'ChessboadHighlighter'
model = keras.Sequential(name='ChessboadHighlighter')

# Input layer with the specified shape
model.add(Input(shape=SHAPE))

# Convolutional layers with max-pooling and dropout
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Transposed convolutional layers with upsampling
model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2DTranspose(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2DTranspose(filters=8, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

# Output layer with sigmoid activation
model.add(Conv2D(filters=1, kernel_size=(3, 3), padding='same', activation='sigmoid'))


# Compile the model with the 'adam' optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display a summary of the model's architecture
model.summary()

# Train the model on the training data with validation data
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val_no_aug, Y_val_no_aug), verbose=2)

# Save the trained model
model.save('../models/chessboard highlighter')


# Evaluating model performance on the test set with and without augmentation
print('Model evaluation on the test set without augmentation:')
print(model.evaluate(X_test_no_aug, Y_test_no_aug, verbose=False))
print('Model evaluation on the test set with augmentation:')
print(model.evaluate(X_test_aug, Y_test_aug, verbose=False))


# Create a figure with two subplots: one for displaying the model's loss and another for its accuracy
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'valation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend(['Train', 'valation'], loc='upper left')

plt.tight_layout()
plt.show()


# Displaying two test set images and their respective model predictions
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X_test_no_aug[0])
axes[0, 0].set_title('X_test_no_aug[0]')

axes[0, 1].imshow(model.predict(np.expand_dims(X_test_no_aug[0], axis=0), verbose=False)[0], cmap='gray')
axes[0, 1].set_title('predict(X_test_no_aug[0])')

axes[1, 0].imshow(X_test_no_aug[1])
axes[1, 0].set_title('X_test_no_aug[1]')

axes[1, 1].imshow(model.predict(np.expand_dims(X_test_no_aug[1], axis=0), verbose=False)[0], cmap='gray')
axes[1, 1].set_title('predict(X_test_no_aug[1])')

plt.tight_layout()
plt.show()


# Define a list of layer names and corresponding images per row
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
images_per_row_list = [8, 8, 16, 16]

# Visualize intermediate activations and filter responses for each layer
for layer_name, images_per_row in zip(layer_names, images_per_row_list):

    visualizeIntermediateActivations(model=model, image=X_test_no_aug[0], output_folder='../files/chessboard detection/c. h. intermediate activations/', layer_name=layer_name, images_per_row=images_per_row)

    visualizeFiltersResponses(model=model, output_folder='../files/chessboard detection/c. h. filters responses/', layer_name=layer_name, images_per_row=images_per_row, input_shape=SHAPE)
