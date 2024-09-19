import numpy as np
import cv2
import time
from agentspace import space, Agent

from rectangleDetection import dominantRectangle, drawRectangle, extractRectangle
from drawingExtraction import extract_trajectories, visualize_trajectories
from imageSimilarity import similarImages

class SeeingAgent(Agent):

    def __init__(self, nameImage, nameFocused, namePicture, nameTrajectories):
        self.nameImage = nameImage
        self.nameFocused = nameFocused
        self.namePicture = namePicture
        self.nameTrajectories = nameTrajectories
        super().__init__()

    def init(self):
        space.attach_trigger(self.nameImage,self)
 
    def senseSelectAct(self):
        
        if space[self.nameTrajectories] is not None: # we have already decided to draw
            return
    
        frame = space[self.nameImage]
        if frame is None:
            return
            
        rect = dominantRectangle(frame)
        if rect is not None:
            print('rect found')
            if space(default=False)[self.nameFocused] or space(default=False)['dontLook']:
                print('focused')
                img = extractRectangle(frame,rect)
                trajectories = extract_trajectories(img)
                if trajectories:
                    candidate = space['candidate']
                    if candidate is None:
                        print('we have a candidate, waiting...')
                        space(validity=3.0)['dontLook'] = True
                        space(validity=3.0)['candidate'] = img
                        time.sleep(0.5)
                    else:
                        print('testing the candidate...')
                        similarity = similarImages(img, candidate)
                        print(f'similarity: {similarity:.2f}')
                        if similarity > 0.3:
                            space[self.namePicture] = img
                            space[self.nameTrajectories] = trajectories
                            space['dontLook'] = False
                            print('we have trajectories')
                            result = visualize_trajectories(trajectories, img.shape)
                            cv2.imshow('picture',result)
                            cv2.waitKey(1)
                        else:
                            print('the candidate failed, trying another one and waiting...')
                            space(validity=3.0)['candidate'] = img
                            time.sleep(0.5)
            drawRectangle(frame,rect)
        
        cv2.imshow('seen',frame)
        cv2.waitKey(1)
        
if __name__ == "__main__":

    from CameraAgent import CameraAgent
    camera_agent = CameraAgent('See3CAM_CU135', 0, 'bgr', fps=30, zoom=350)
    space['focused'] = True
    seeing_agent = SeeingAgent('bgr', 'focused', 'picture', 'trajectories')
    input()
    seeing_agent.stop()
    camera_agent.stop()
