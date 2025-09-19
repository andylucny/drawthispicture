import keras
keras.__version__
from keras.models import load_model

model = load_model('perceptron.h5')
model.summary()

import tensorflow as tf
import tf2onnx
# Convert the Keras model to ONNX
spec = (tf.TensorSpec((None, *model.input.shape[1:]), tf.float32, name="input"),)
output_path = "perceptron.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
