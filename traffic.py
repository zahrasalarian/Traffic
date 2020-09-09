import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    images = []
    labels = []
    os.chdir(r"{}".format(data_dir))
    files = os.listdir()
    # print(files)
    for file in files:
        os.chdir(r"{}".format(file))
        fs = os.listdir()
        for f in fs:
            img = cv2.imread(f)
            # np.reshape(img,(-1,IMG_HEIGHT,3))
            img  = cv2.resize(img, dsize=(IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
            print(img.shape)
            images.append(img)
            labels.append(file)
        # print(file)
        os.chdir(r"..")
    images = np.asarray(images)
    labels = np.asarray(labels)

    images = np.expand_dims(images,-1)
    labels = np.expand_dims(labels,-1)
    images = np.asarray(images).astype(np.float32)
    labels = np.asarray(labels).astype(np.float32)
    return (images,labels)
    # raise NotImplementedError


def get_model():
    # Create a convolutional neural network
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(30,30,3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),

        # Add hidden layer
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        # Add output layers with output units for all 43 categories
        tf.keras.layers.Dense(43,activation="softmax")
    ])
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"])
    return model
    # raise NotImplementedError


if __name__ == "__main__":
    main()
