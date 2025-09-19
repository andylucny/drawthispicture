import keras
from keras.models import load_model
import numpy as np

model = load_model('perceptron.h5')

sample_output = np.array([22.1, 42.4, 17.6, 126.3, 163.7, 122.5, -179.9],np.float32)
blob = np.array([[0.4204166531562805, 0.5814814567565918]],np.float32)
output = model(blob)[0]*180

print(output-sample_output)