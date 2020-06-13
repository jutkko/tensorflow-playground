import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
ys = np.array([-8.0, -1.0, 0.0, 1.0, 8.0, 27.0, 64.0, 125.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print(model.predict([11.0]))
print(model.predict([100.0]))