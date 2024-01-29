import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def create_autoencoder():
    autoencoder = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Encoder
        tf.keras.layers.Dense(64, activation='relu'),   # Latent space
        tf.keras.layers.Dense(128, activation='relu'),  # Decoder
        tf.keras.layers.Dense(28 * 28, activation='sigmoid'),  # Output layer (reconstructed image)
        tf.keras.layers.Reshape((28, 28))  # Reshape to match input shape
    ])
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder


def create_classifier():
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer for classification (10 classes for digits)
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier


def visualize_encoder_results(autoencoder, x_test):
    # Visualize original and reconstructed images
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


def main():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

    print('Create and train the autoencoder')
    autoencoder = create_autoencoder()
    autoencoder.fit(x_train, x_train, epochs=5)

    print('Visualize the encoder results')
    visualize_encoder_results(autoencoder, x_test)

    print('\nCreate and train the classifier')
    classifier = create_classifier()
    classifier.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    print('Evaluate the classifier on the test set')
    classifier.evaluate(x_test, y_test)

    print('\nPredict the first 5 images in the test set using trained classifier')
    predictions = classifier.predict(x_test[:5])
    print('Test values: ', y_test[:5])
    print('Predicted values:', np.argmax(predictions, axis=1))


if __name__ == "__main__":
    main()
