print("importing cv2")
import cv2
print("importing numpy")
import numpy as np
print("importing os and sys")
import os
import sys
print("importing tensorflow")
import tensorflow as tf

print("importing train_test_split")
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    print("main")

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    print("main 2")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    print("main 3")
    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    print("labels.shape", labels.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    print("main 4")
    # Get a compiled neural network
    model = get_model()

    print("main 5")
    model.summary()
    
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    print("main 6")
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    print("main 7")
    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.
    
    Use the OpenCV-Python module (cv2) to read each image as a numpy.ndarray (a numpy multidimensional array). To pass these images into a neural network, the images will need to be the same size, so be sure to resize each image to have width IMG_WIDTH and height IMG_HEIGHT.

    Your function should be platform-independent: that is to say, it should work regardless of operating system. Note that on macOS, the / character is used to separate path components, while the \ character is used on Windows. Use os.sep and os.path.join as needed instead of using your platform’s specific separator character.
    
    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            continue

        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)

            image = cv2.imread(file_path)
            if image is not None:
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(category)

    return images, labels
    

def get_model():
    """
    Returns a compiled convolutional neural network model.
    
    You may assume that the input to the neural network will be of the shape (IMG_WIDTH, IMG_HEIGHT, 3) (that is, an array representing an image of width IMG_WIDTH, height IMG_HEIGHT, and 3 values for each pixel for red, green, and blue).

    The output layer of the neural network should have NUM_CATEGORIES units, one for each of the traffic sign categories.

    The number of layers and the types of layers you include in between are up to you. You may wish to experiment with:
        different numbers of convolutional and pooling layers
        different numbers and sizes of filters for convolutional layers
        different pool sizes for pooling layers
        different numbers and sizes of hidden layers
        dropout
    """

    # Initialize a sequential model
    model = tf.keras.models.Sequential()

    # First convolutional layer with pooling
    # This layer detects basic features in the image with filters
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer with pooling
    # This layer detects more complex patterns
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Add a dropout layer to reduce overfitting
    model.add(tf.keras.layers.Dropout(0.5))

    # Flatten the output to connect it to the fully connected layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected (dense) layer for pattern recognition
    model.add(tf.keras.layers.Dense(128, activation="relu"))

    # Output layer with units for each category
    # Softmax is used to predict the probability for each category
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"))

    # Compile the model with an optimizer, loss function, and evaluation metric
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
