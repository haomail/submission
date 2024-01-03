# =============================================================================
# PROBLEM B1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-3
# =============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras


def solution_B1():
    # DO NOT CHANGE THIS CODE
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    Y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    # YOUR CODE HERE
    normalizer = tf.keras.layers.Normalization(axis=None, input_shape=(1,))
    normalizer.adapt(X)

    # Define the model
    model = keras.Sequential()
    model.add(normalizer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    # Define a callback to stop training when MSE is lower than 1e-3
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('loss') < 1e-3):
                print("\nMSE is lower than 1e-3!")
                self.model.stop_training = True

    # Instantiate the callback
    loss_callback = Callback()

    # Compile the model
    model.compile(optimizer='adam',
                  loss= 'mean_squared_error',
                  metrics=['accuracy'] )

    # Train the model
    model.fit(X, Y, epochs=1000, callbacks=[loss_callback])
    print(model.predict([-2.0, 10.0]))
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B1()
    model.save("model_B1.h5")
