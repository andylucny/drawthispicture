import os
import io
import requests
import zipfile
import numpy as np
import cv2

def download_zipfile(path,url):
    if os.path.exists(path):
        return
    print("downloading",url)
    response = requests.get(url)
    if response.ok:
        file_like_object = io.BytesIO(response.content)
        zipfile_object = zipfile.ZipFile(file_like_object)    
        zipfile_object.extractall(".")
    print("downloaded")
    
def download_hed():
    download_zipfile('hed_pretrained_bsds.caffemodel','http://www.agentspace.org/download/hed.zip')

download_hed()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
 
    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
 
        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width
 
        return [[batchSize, numChannels, height, width]]
 
    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv2.dnn_registerLayer('Crop', CropLayer)
        
face_architecture = 'hed_pretrained_bsds.prototxt'
face_weights = 'hed_pretrained_bsds.caffemodel'
net = cv2.dnn.readNetFromCaffe(face_architecture, face_weights)

def deepEdges(frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    out = out[0, 0]
    edges = cv2.resize(out, (frame.shape[1], frame.shape[0]))
    edges = 255 * edges
    edges = edges.astype(np.uint8)
    return edges
       
if __name__ == '__main__':
    frame = cv2.imread('1724513323.png')
    edges = deepEdges(frame)
    cv2.imwrite('edges.png',edges)
