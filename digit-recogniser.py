import os
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

print(tf.__version__)

def load_data():
	# Keras provides a handy API to download the MNIST dataset, and split them into
	# "train" dataset and "test" dataset.
	mnist = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	# Normalize the input image so that each pixel value is between 0 to 1.
	train_images = train_images / 255.0
	test_images = test_images / 255.0
	print('Pixels are normalized')
	return train_images, train_labels, test_images, test_labels

def draw_samples(images, labels):
	# Show the first 25 images in the training dataset.
	plt.figure(figsize=(10,10))
	for i in range(25):
	  plt.subplot(5,5,i+1)
	  plt.xticks([])
	  plt.yticks([])
	  plt.grid(False)
	  plt.imshow(images[i], cmap=plt.cm.gray)
	  plt.xlabel(labels[i])

	plt.show()

# An utility function that returns where the digit is in the image.
def digit_area(mnist_image):
	# Remove the color axes
	# mnist_image = np.squeeze(mnist_image, axis=2)

	# Extract the list of columns that contain at least 1 pixel from the digit
	x_nonzero = np.nonzero(np.amax(mnist_image, 0))
	x_min = np.min(x_nonzero)
	x_max = np.max(x_nonzero)

	# Extract the list of rows that contain at least 1 pixel from the digit
	y_nonzero = np.nonzero(np.amax(mnist_image, 1))
	y_min = np.min(y_nonzero)
	y_max = np.max(y_nonzero)

	return [x_min, x_max, y_min, y_max]

def show_histogram(images):
	# Calculate the area containing the digit across MNIST dataset
	digit_area_rows = []
	for image in images:
		digit_area_row = digit_area(image)
		digit_area_rows.append(digit_area_row)
		digit_area_df = pd.DataFrame(digit_area_rows,
		columns=['x_min', 'x_max', 'y_min', 'y_max'])
	digit_area_df.hist()
	plt.show()

# Define a simple sequential model
def create_model():
	# Define the model architecture
	model = keras.Sequential([
		keras.layers.InputLayer(input_shape=(28, 28)),
		keras.layers.Reshape(target_shape=(28, 28, 1)),
		keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu),
		keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu),
		keras.layers.MaxPooling2D(pool_size=(2, 2)),
		keras.layers.Dropout(0.25),
		keras.layers.Flatten(),
		keras.layers.Dense(10)
	])
	# Define how to train the model
	model.compile(optimizer='adam',
				  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])
	return model

def fit(model, images, labels):
	checkpoint_path = "training_1/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
	                                                 save_weights_only=True,
	                                                 verbose=1,
	                                                 period=1)

	# Train the digit classification model
	model.fit(images, labels, epochs=1, callbacks=[cp_callback])

def augument_data_generator(train_images, train_labels, test_images, test_labels):
	# Define data augmentation
	datagen = keras.preprocessing.image.ImageDataGenerator(
		rotation_range=30,
		width_shift_range=0.25,
		height_shift_range=0.25,
		shear_range=0.25,
		zoom_range=0.2
	)

	# Generate augmented data from MNIST dataset
	train_generator = datagen.flow(train_images, train_labels)
	test_generator = datagen.flow(test_images, test_labels)
	return train_generator, test_generator

train_images, train_labels, test_images, test_labels = load_data()

# Add a color dimension to the images in "train" and "validate" dataset to
# leverage Keras's data augmentation utilities later.
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

train_generator, test_generator = augument_data_generator(train_images, train_labels, test_images, test_labels)

augmented_images, augmented_labels = next(train_generator)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(augmented_images[i], axis=2), cmap=plt.cm.gray)
    plt.xlabel('Label: %d' % augmented_labels[i])
plt.show()
# draw_samples(train_images, train_labels)

# show_histogram(test_images)
# Create a new model instance
# model = create_model()

# fit(model, train_images, train_labels)
# model.summary()

# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)

# # A helper function that returns 'red'/'black' depending on if its two input
# # parameter matches or not.
# def get_label_color(val1, val2):
#   if val1 == val2:
#     return 'black'
#   else:
#     return 'red'

# # Predict the labels of digit images in our test dataset.
# predictions = model.predict(test_images)

# # As the model output 10 float representing the probability of the input image
# # being a digit from 0 to 9, we need to find the largest probability value
# # to find out which digit the model predicts to be most likely in the image.
# prediction_digits = np.argmax(predictions, axis=1)

# # Then plot 100 random test images and their predicted labels.
# # If a prediction result is different from the label provided label in "test"
# # dataset, we will highlight it in red color.
# plt.figure(figsize=(18, 18))
# for i in range(100):
#   ax = plt.subplot(10, 10, i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   image_index = random.randint(0, len(prediction_digits))
#   plt.imshow(test_images[image_index], cmap=plt.cm.gray)
#   ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index],\
#                                            test_labels[image_index]))
#   plt.xlabel('Predicted: %d' % prediction_digits[image_index])
# plt.show()