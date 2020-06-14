import os
import tensorflow as tf
from tensorflow import keras

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
test_images = test_images / 255.0

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

# Create a new model instance
model = create_model()
# Load the previously saved weights
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

model.load_weights(latest)

model.save('training_1/my_model')
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)