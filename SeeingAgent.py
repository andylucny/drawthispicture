import numpy as np
import cv2
from agentspace import space, Agent

from rectangleDetection import dominantRectangle, drawRectangle, extractRectangle
from drawingExtraction import extract_trajectories, visualize_trajectories

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
        frame = space[self.nameImage]
        if frame is None:
            return
            
        rect = dominantRectangle(frame)
        if rect is not None:
            if space(default=False)[self.nameFocused]:
                img = extractRectangle(frame,rect)
                trajectories = extract_trajectories(img)
                if trajectories:
                    space(validity = 1.0)[self.namePicture] = img
                    space(validity = 1.0)[self.nameTrajectories] = trajectories

                result = visualize_trajectories(trajectories, img.shape)
                cv2.imshow('picture',result)

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
