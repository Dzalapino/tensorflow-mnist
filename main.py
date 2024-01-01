def main():
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist

    # Print the version of TensorFlow
    print("TensorFlow version: {}".format(tf.__version__))

    # Set the logging level to debug
    tf.get_logger().setLevel('FATAL')

    # Load the data and split it into training and testing sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize the data (scale it to 0-1) and convert it to float32
    x_train, x_test = tf.cast(x_train / 255.0, tf.float32), tf.cast(x_test / 255.0, tf.float32)

    # Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the training set
    model.fit(x_train, y_train, epochs=5)

    # Evaluate the model on the test set
    model.evaluate(x_test, y_test)

    # Show first 5 images in the test set
    for i in range(5):
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.show()

    # Print the labels of the first 5 images in the test set
    print(f'\nfirst 5 images in the test set labels:\n{y_test[:5]}')

    # Set the format of the predictions to 3 decimal places
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

    # Predict the first 5 images in the test set
    predictions = model.predict(x_test[:5])

    # Show the predictions of the first 5 images in the test set as probabilities for each class
    for i in range(5):
        print(f'Prediction for image {i}:\n{predictions[i]}\n')


if __name__ == "__main__":
    main()
