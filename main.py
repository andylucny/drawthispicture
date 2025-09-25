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
from TouchAgent import TouchAgent
from DrawingAgent import DrawingAgent
from LipsAgent import LipsAgent
from ListenerAgent import ListenerAgent
from TranscriptionAgent import TranscriptionAgent
from ResponderAgent import ResponderAgent

CameraAgent('See3CAM_CU135',1,'robotEye',fps=10,zoom=350) # right eye
time.sleep(1)
PerceptionAgent('robotEye','clipFeatures','dinoPoints')
time.sleep(1)
LookAroundAgent('dinoPoints','dontLook','focused')
time.sleep(1)
SpeakerAgent('tospeak')
time.sleep(1)
NamingAgent('clipFeatures', 'focused', 'picture')
time.sleep(1)
SeeingAgent('robotEye', 'focused', 'picture', 'trajectories')
time.sleep(1)
TouchAgent()
time.sleep(1)
DrawingAgent('trajectories')
time.sleep(1)
LipsAgent() # move with lips
time.sleep(1)
ListenerAgent('audio',1) #2 # listen to audio
time.sleep(1)
TranscriptionAgent('audio','text') # transcribe audio into text
time.sleep(1)
ResponderAgent('text', 'picture', 'trajectories') # respond to the queries
time.sleep(1)

def en():
    space['language'] = 'en'

def sk():
    space['language'] = 'sk'

def suspend():
    space(priority=10)['robotEye'] = np.zeros((480,640,3),np.uint8)

def resume():
    space(priority=10)['robotEye'] = None
    space(validity=0.1)['tospeak'] = 'Ideme na to!' if space(default='en')['language'] == 'sk' else "Let's go interacting!"

time.sleep(2)
resume()

