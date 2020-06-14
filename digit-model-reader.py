import os
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = tf.keras.models.load_model('training_1/my_model')

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)