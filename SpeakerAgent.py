import os
import sys
import time
from agentspace import Agent, space, Trigger
from speak import speak

class SpeakerAgent(Agent):

    def __init__(self, nameText):
        self.nameText = nameText
        super().__init__()
        
    def init(self):
        space.attach_trigger(self.nameText,self,Trigger.NAMES_AND_VALUES)
        
    def senseSelectAct(self):
        _, text = self.triggered()
        speak(text)

if __name__ == "__main__":  
    text = sys.argv[1] if len(sys.argv) > 1 else "hallo"
    space['language'] = 'cz'
    speak(text)
    time.sleep(1)
    agent = SpeakerAgent('text')
    time.sleep(1)
    text = 'toto poviem tri razy' if space['language'] == 'sk' else 'tohle řeknu třikrát' if space['language'] == 'cz' else 'three times speaking' 
    space['text']= text
    space['text']= text
    space['text']= text
    time.sleep(10)
    agent.stop()
