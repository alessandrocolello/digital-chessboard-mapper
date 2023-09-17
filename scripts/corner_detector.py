# Importing used libraries
from utils import loadImagesWithKeypoints
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np


# Clearing session and setting random seed
keras.backend.clear_session()
keras.utils.set_random_seed(42)


# Loading Chessboard Highlighter
ChessboardHighlighter = keras.models.load_model('../models/chessboard highlighter')


# Defining constants
NUM_KEYPOINTS = 4
SHAPE = [224, 224, 1] # [height, width, channels]
BATCH_SIZE = 16
EPOCHS = 60
dataset_folder = '../files/chessboard detection/dataset/'


# Loading raw training, validation, and test sets
X_train, y_train = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='train')
X_val_no_aug, y_val_no_aug, X_val_aug, y_val_aug = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='val', split=True)
X_test_no_aug, y_test_no_aug, X_test_aug, y_test_aug = loadImagesWithKeypoints(dataset_folder=dataset_folder, set_to_load='test', split=True)


# Predicting chessboard masks of training, validation, and test images
X_train = ChessboardHighlighter.predict(X_train, verbose=False)
X_val_no_aug = ChessboardHighlighter.predict(X_val_no_aug, verbose=False)
X_val_aug = ChessboardHighlighter.predict(X_val_aug, verbose=False)
X_test_no_aug = ChessboardHighlighter.predict(X_test_no_aug, verbose=False)
X_test_aug = ChessboardHighlighter.predict(X_test_aug, verbose=False)


# Displaying four training images with their respective keypoints
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X_train[0], cmap='gray')
axes[0, 0].set_title('X_train[0]')
axes[0, 0].scatter(y_train[0, [0, 2, 4, 6]]*SHAPE[1], y_train[0, [1, 3, 5, 7]]*SHAPE[0])

axes[0, 1].imshow(X_train[1], cmap='gray')
axes[0, 1].set_title('X_train[1]')
axes[0, 1].scatter(y_train[1, [0, 2, 4, 6]]*SHAPE[1], y_train[1, [1, 3, 5, 7]]*SHAPE[0])

axes[1, 0].imshow(X_train[2], cmap='gray')
axes[1, 0].set_title('X_train[2]')
axes[1, 0].scatter(y_train[2, [0, 2, 4, 6]]*SHAPE[1], y_train[2, [1, 3, 5, 7]]*SHAPE[0])

axes[1, 1].imshow(X_train[3], cmap='gray')
axes[1, 1].set_title('X_train[3]')
axes[1, 1].scatter(y_train[3, [0, 2, 4, 6]]*SHAPE[1], y_train[3, [1, 3, 5, 7]]*SHAPE[0])

plt.tight_layout()
plt.show()


# Create a Sequential model with the name 'CornerDetector'
model = keras.Sequential(name='CornerDetector')

# Input layer with the specified shape
model.add(Input(shape=SHAPE))

# Convolutional layers with max-pooling
for filters in [8, 16]:

    model.add(Conv2D(filters=filters, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

for filters in [32, 64]:

    model.add(Conv2D(filters=filters, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the convolutional layers
model.add(Flatten())

# Dropout layer
model.add(Dropout(0.5))

# Fully connected layer with 256 units and sigmoid activation
model.add(Dense(units=128, activation='relu'))

# Output layer with sigmoid activation
model.add(Dense(units=2*NUM_KEYPOINTS, activation='sigmoid'))


# Compile the model with the 'adam' optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mse')

# Display a summary of the model's architecture
model.summary()


# Train the model on the training data with validation data
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val_no_aug, y_val_no_aug), verbose=2)

# Save the trained model
model.save('../models/corner detector')


# Evaluating model performance on the test set with and without augmentation
print('Model evaluation on original test set images:')
print(model.evaluate(X_test_no_aug, y_test_no_aug, verbose=False))
print('Model evaluation on augmented test set images:')
print(model.evaluate(X_test_aug, y_test_aug, verbose=False))


# Displaying four test not augmented images with their respective actual and predicted keypoints
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X_test_no_aug[0], cmap='gray')
axes[0, 0].set_title('X_test_no_aug[0]')
axes[0, 0].scatter(y_test_no_aug[0, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[0, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[0], axis=0), verbose=False)[0]
axes[0, 0].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[0, 1].imshow(X_test_no_aug[1], cmap='gray')
axes[0, 1].set_title('X_test_no_aug[1]')
axes[0, 1].scatter(y_test_no_aug[1, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[1, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[1], axis=0), verbose=False)[0]
axes[0, 1].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[1, 0].imshow(X_test_no_aug[2], cmap='gray')
axes[1, 0].set_title('X_test_no_aug[2]')
axes[1, 0].scatter(y_test_no_aug[2, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[2, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[2], axis=0), verbose=False)[0]
axes[1, 0].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[1, 1].imshow(X_test_no_aug[3], cmap='gray')
axes[1, 1].set_title('X_test_no_aug[3]')
axes[1, 1].scatter(y_test_no_aug[3, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[3, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[3], axis=0), verbose=False)[0]
axes[1, 1].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

plt.tight_layout()
plt.show()


# Displaying four test not augmented images with their respective actual and predicted keypoints
fig, axes = plt.subplots(2, 2)

axes[0, 0].imshow(X_test_no_aug[5], cmap='gray')
axes[0, 0].set_title('X_test_no_aug[5]')
axes[0, 0].scatter(y_test_no_aug[5, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[5, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[5], axis=0), verbose=False)[0]
axes[0, 0].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[0, 1].imshow(X_test_no_aug[6], cmap='gray')
axes[0, 1].set_title('X_test_no_aug[6]')
axes[0, 1].scatter(y_test_no_aug[6, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[6, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[6], axis=0), verbose=False)[0]
axes[0, 1].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[1, 0].imshow(X_test_no_aug[7], cmap='gray')
axes[1, 0].set_title('X_test_no_aug[7]')
axes[1, 0].scatter(y_test_no_aug[7, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[7, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[7], axis=0), verbose=False)[0]
axes[1, 0].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

axes[1, 1].imshow(X_test_no_aug[8], cmap='gray')
axes[1, 1].set_title('X_test_no_aug[8]')
axes[1, 1].scatter(y_test_no_aug[8, [0, 2, 4, 6]]*SHAPE[1], y_test_no_aug[8, [1, 3, 5, 7]]*SHAPE[0])
pred = model.predict(np.expand_dims(X_test_no_aug[8], axis=0), verbose=False)[0]
axes[1, 1].scatter(pred[[0, 2, 4, 6]]*SHAPE[1], pred[[1, 3, 5, 7]]*SHAPE[0])

plt.tight_layout()
plt.show()


# Plotting the model's loss on training and validation sets
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MSE')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

