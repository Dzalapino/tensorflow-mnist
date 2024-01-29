import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def create_autoencoder() -> tf.keras.models.Sequential:
    """
    Create an autoencoder model using the Sequential API
    :return: autoencoder model
    """
    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Encoder
        tf.keras.layers.Dense(64, activation='relu'),  # Latent space
        tf.keras.layers.Dense(128, activation='relu'),  # Decoder
        tf.keras.layers.Dense(28 * 28, activation='sigmoid'),  # Output layer (reconstructed image)
        tf.keras.layers.Reshape((28, 28))  # Reshape to match input shape
    ])
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder


def create_classifier() -> tf.keras.models.Sequential:
    """
    Create a classifier model using the Sequential API for classifying MNIST digits.
    :return: classifier model
    """
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer for classification (10 classes for digits)
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier


def visualize_encoder_results(autoencoder, x_test) -> None:
    """
    Visualize the encoder results using matplotlib. The encoder results are the reconstructed images.
    :param autoencoder: autoencoder model
    :param x_test: test images to visualize
    :return: None
    """
    reconstructed_images = autoencoder.predict(x_test)

    plt.figure(figsize=(10, 4))
    for i in range(5):
        # Original Image
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title("Original")

        # Reconstructed Image
        plt.subplot(2, 5, i + 6)
        plt.imshow(reconstructed_images[i], cmap='gray')
        plt.title("Reconstructed")
    plt.show()


def visualize_training_history(history) -> None:
    """
    Visualize the training history using matplotlib.
    :param history: training history to visualize
    :return: None
    """
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='lower right')
    plt.show()


def visualize_model_predictions(num_rows: int, num_cols: int, test_images, predictions, test_labels) -> None:
    """
    Visualize the model predictions on the test dataset using matplotlib.
    Number of images to display is num_rows * num_cols.
    :param num_rows: number of rows to display
    :param num_cols: number of columns to display
    :param test_images: test images to display
    :param predictions: predictions to display
    :param test_labels: test labels to display
    :return: None
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        predicted_label = np.argmax(predictions[i])
        true_label = test_labels[i]
        if predicted_label == true_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)
    plt.show()


def main():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    print('Create and train the autoencoder')
    autoencoder = create_autoencoder()
    history = autoencoder.fit(x_train, x_train, epochs=5, validation_data=(x_val, x_val))

    print('\nVisualize the training history for the autoencoder')
    visualize_training_history(history)

    print('Visualize the encoder results')
    visualize_encoder_results(autoencoder, x_test)

    print('\nCreate and train the classifier')
    classifier = create_classifier()
    history = classifier.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

    print('\nVisualize the training history for the classifier')
    visualize_training_history(history)

    print('Evaluate the classifier on the test set')
    classifier.evaluate(x_test, y_test)

    print('\nPredict the first 25 images in the test set using trained classifier')
    predictions = classifier.predict(x_test[:25])

    print('Visualize the model predictions')
    visualize_model_predictions(5, 5, x_test, predictions, y_test)


if __name__ == "__main__":
    main()
