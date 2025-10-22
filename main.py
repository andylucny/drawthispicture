import time
import os
import signal
import re
import numpy as np
import cv2
from agentspace import Agent, space, Trigger

def quit():
    os._exit(0)

def signal_handler(signal, frame): 
    quit()

signal.signal(signal.SIGINT, signal_handler)

from config import transcription

from CameraAgent import CameraAgent
from PerceptionAgent import PerceptionAgent
from LookAroundAgent import LookAroundAgent
from SpeakerAgent import SpeakerAgent
from NamingAgent import NamingAgent
from SeeingAgent import SeeingAgent
from TouchAgent import TouchAgent
from DrawingAgent import DrawingAgent
from LipsAgent import LipsAgent
if transcription:
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
NamingAgent('clipFeatures', 'focused', 'picture', clip_threshold=0.15, judgement_threshold=0.25)
time.sleep(1)
SeeingAgent('robotEye', 'focused', 'picture', 'trajectories')
time.sleep(1)
TouchAgent()
time.sleep(1)
DrawingAgent('trajectories')
time.sleep(1)
LipsAgent() # move with lips
time.sleep(1)
if transcription:
    ListenerAgent('audio',1) #3 for Jabra, 1 for ATR or IO2 # listen to audio
    time.sleep(1)
    TranscriptionAgent('audio','text') # transcribe audio into text via whisper
    time.sleep(1)
ResponderAgent('text', 'picture', 'trajectories') # respond to the queries
time.sleep(1)

def en():
    space['language'] = 'en'

def sk():
    space['language'] = 'sk'

def cz():
    space['language'] = 'cz'
    
def suspend():
    space(priority=10)['robotEye'] = np.zeros((480,640,3),np.uint8)

def resume():
    space(priority=10)['robotEye'] = None
    lang = space(default='en')['language']
    space(validity=0.1)['tospeak'] = 'Ideme na to!' if lang == 'sk' else 'Jdeme na to!' if lang == 'cz' else "Let's go interacting!"

def enter(text):
    space['text'] = text

def draw(text):
    space['text'] = "Nakresli mi " + text

def save(count=1):
    for _ in range(count):
        time.sleep(1)
        image = space['robotEye']
        if image is not None:
            image_file = str(int(time.time()))+'.png'
            cv2.imwrite(image_file,image)
            print(image_file,'recorded')

time.sleep(2)
cz()
resume()

