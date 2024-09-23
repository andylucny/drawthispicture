import time
import numpy as np
from agentspace import space, Agent

from drawingExecution import draw_trajectories
from TouchAgent import clear

class DrawingAgent(Agent):

    def __init__(self, nameTrajectories):
        self.nameTrajectories = nameTrajectories
        super().__init__()

    def init(self):
        space.attach_trigger(self.nameTrajectories,self)

    def speak(self, text):
        space(validity=0.1)['tospeak'] = text
        
    def senseSelectAct(self):
        trajectories = space[self.nameTrajectories]
        if trajectories is None:
            return

        space(priority=5)['dontLook'] = True
        clear()
        draw_trajectories(trajectories)
        space(priority=5)['dontLook'] = None
            
        if space(default='en')['language'] == 'sk':
            text = 'Hotovo. Dáme si oddych.'
        else:
            text = "Done. Let's take a break."
        self.speak(text)  
        
        space['picture'] = None
        print("printing takes a rest for one minute from now")
        time.sleep(60)
        space[self.nameTrajectories] = None

        if space(default='en')['language'] == 'sk':
            text = 'Dobre.'
        else:
            text = "Ready."
        self.speak(text)  
        print("printing activated again")
        
if __name__ == "__main__":
    
    drawing_agent = DrawingAgent('trajectories')
    time.sleep(4)
    #drawing_agent.stop()
