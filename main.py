import time
import os
import signal
import re
import numpy as np
import cv2 as cv
from agentspace import Agent, space, Trigger

def quit():
    os._exit(0)

def signal_handler(signal, frame): 
    quit()
    
signal.signal(signal.SIGINT, signal_handler)

from CameraAgent import CameraAgent
from PerceptionAgent import PerceptionAgent
from LookAroundAgent import LookAroundAgent
from SpeakerAgent import SpeakerAgent
from NamingAgent import NamingAgent
from SeeingAgent import SeeingAgent
#from DrawingAgent import DrawingAgent

CameraAgent('See3CAM_CU135',1,'robotEye',fps=10,zoom=350) # right eye
time.sleep(1)
PerceptionAgent('robotEye','clipFeatures','dinoPoints')
time.sleep(1)
LookAroundAgent('dinoPoints','dontLook','focused')
time.sleep(1)
SpeakerAgent('tospeak')
time.sleep(1)
NamingAgent('clipFeatures', 'focused')
time.sleep(1)
SeeingAgent('robotEye', 'focused', 'picture', 'trajectories')
time.sleep(1)
#DrawingAgent('picture', 'dinoPoints')
#time.sleep(1)

def en():
    space['language'] = 'en'

def sk():
    space['language'] = 'sk'
