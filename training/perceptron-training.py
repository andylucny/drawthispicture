import numpy as np
import keras
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import cv2
from nicomover import load_movement

ds = load_movement('dataset.txt')
indices = ['r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x', 'r_indexfinger_x', 'x', 'y']
ds = np.array([[sample[index] for index in indices] for sample in ds])
samples_inp = ds[:,-2:]
samples_out = ds[:,:-2]/180.0

inp = Input(shape=(2,))
x = Dense(50,activation='relu')(inp)
out = Dense(7,activation='linear')(x)
model = Model(inputs=inp, outputs=out)
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

model.fit(samples_inp, samples_out, batch_size=10, epochs=30000)

model.save('perceptron.h5')

# test
results = model.predict(samples_inp)
for sample, result in zip(samples_out,results):
    print([int(value) for value in list(sample*180)+list(result*180)])

resolution = (2400, 1350)
x = np.linspace(0,1,resolution[0])
y = np.linspace(0,1,resolution[1])
xx, yy = np.meshgrid(x,y)
inps = np.array([xx.reshape(-1),yy.reshape(-1)]).T
outs = model.predict(inps)
for i, out in enumerate(outs.T):
    vals = np.asarray(out.reshape((resolution[1],resolution[0])),np.float32)
    cv2.imwrite(str(i)+'.tif',vals)
    disp = np.asarray(out.reshape((resolution[1],resolution[0]))*127+127,np.uint8)
    cv2.imwrite(str(i)+'.png',disp)
    
    