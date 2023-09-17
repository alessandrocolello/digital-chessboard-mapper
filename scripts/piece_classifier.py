# Importing used libraries
import numpy as np
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, SGD
from keras.regularizers import L2
from utils import loadChessPieceImages, visualizeIntermediateActivations, visualizeFiltersResponses, visualizeHeatmap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Clearing session and setting random seed
keras.backend.clear_session()
keras.utils.set_random_seed(42)


# Defining constants
SHAPE = [64, 64, 3]
BATCH_SIZE = 32
EPOCHS = 60
dataset_folder = '../files/piece classification/dataset/'


# Loading training, validation, and test sets
X_train, y_train = loadChessPieceImages(dataset_folder=dataset_folder, set_to_load='train')
X_val_no_aug, y_val_no_aug, X_val_aug, y_val_aug = loadChessPieceImages(dataset_folder=dataset_folder, set_to_load='val', split=True)
X_test_no_aug, y_test_no_aug, X_test_aug, y_test_aug = loadChessPieceImages(dataset_folder=dataset_folder, set_to_load='test', split=True)

# Displaying training images
subplots_per_row = 3
fig, axes = plt.subplots(subplots_per_row, subplots_per_row)

for i in range(subplots_per_row):
    for j in range(subplots_per_row):

        axes[i, j].imshow(X_train[subplots_per_row*i+j], cmap='gray')
        axes[i, j].set_title(str(np.argmax(y_train[subplots_per_row*i+j])))
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()


# Create a Sequential model with the name 'PieceClassifier'
model = keras.Sequential(name='PieceClassifier')

# Input layer with the specified shape
model.add(Input(shape=SHAPE))

# Convolutional layers with dropout and max-pooling
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Dropout layer
model.add(Dropout(0.5))

# Fully connected layer with 128 units, batch normalization and relu activation
model.add(Dense(units=64, kernel_regularizer=L2(0.001)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Output layer
model.add(Dense(units=13, activation='softmax', kernel_regularizer=L2(0.001)))


# Compile the model with the 'adam' optimizer, categorical crossentropy loss, and accuracy as the evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display a summary of the model's architecture
model.summary()

# Train the model on the training data with validation data
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val_no_aug, y_val_no_aug), verbose=2)

# Save the trained model
model.save('../models/piece classifier')


# Plot training & validation loss values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Metric')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')

plt.tight_layout()
plt.show()


# Evaluating model performance on the test set with and without augmentation
print('Model evaluation on original test set images:')
print(model.evaluate(X_test_no_aug, y_test_no_aug, verbose=False))
print('Model evaluation on augmented test set images:')
print(model.evaluate(X_test_aug, y_test_aug, verbose=False))


# Define a list of layer names and corresponding images per row
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
images_per_row_list = [8, 8, 16, 16]
 
# Visualize intermediate activations and filter responses for each layer
for layer_name, images_per_row in zip(layer_names, images_per_row_list):

    visualizeIntermediateActivations(model=model, image=X_test_no_aug[0], output_folder='../files/piece classification/intermediate activations/', layer_name=layer_name, images_per_row=images_per_row)

    visualizeFiltersResponses(model=model, output_folder='../files/piece classification/filters/', layer_name=layer_name, images_per_row=images_per_row, input_shape=SHAPE)


# Define a list of class names
classes = ['Empty Square', 'White Pawn', 'White Knight', 'White Bishop', 'White Rook', 'White Queen', 'White King', 'Black Pawn', 'Black Knight', 'Black Bishop', 'Black Rook', 'Black Queen', 'Black King']
j = 0

# Visualize heatmaps for one image of each class
for i in np.argmax(y_test_no_aug, axis=0): 
    
    preds = model.predict(np.expand_dims(X_test_no_aug[i], axis=0), verbose=False)
    
    visualizeHeatmap(model=model, last_conv_layer_name='conv2d', image=X_test_no_aug[i], pred_index=np.argmax(preds), 
                     output_path='../files/piece classification/heatmaps/'+classes[j]+'.png', alpha=0.01)

    j += 1


# Define a list of piece labels
pieces = ['E', 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

# Predict and visualize confusion matrix
y_pred = model.predict(X_test_no_aug, verbose=False)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test_no_aug, axis=1)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pieces).plot()
plt.show()
