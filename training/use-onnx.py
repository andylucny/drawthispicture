import numpy as np
import onnxruntime as ort
import cv2 as cv

session = ort.InferenceSession(f"perceptron.onnx", providers=['CPUExecutionProvider'])
input_names = [input.name for input in session.get_inputs()]
output_names = [output.name for output in session.get_outputs()]

sample_output = np.array([22.1, 42.4, 17.6, 126.3, 163.7, 122.5, -179.9],np.float32)
blob = np.array([[0.4204166531562805, 0.5814814567565918]],np.float32)
      
data_input = { input_names[0] : blob }
data_output = session.run(output_names, data_input)
output = data_output[0]*180

print(output-sample_output)
