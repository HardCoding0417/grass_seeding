from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_docs.vis import embed
import tensorflow as tf
import numpy as np
import imageio

batch_size = 64
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train)

# all_digits = np.concatenate([x_train, x_test])
# all_labels = np.concatenate([y_train, y_test])
#
# all_digits = all_digits.astype("float32") / 255.0
# all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
# all_labels = keras.utils.to_categorical(all_labels, 10)
#
# dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
# dataset = dataset.shuffle(buffer_size)

