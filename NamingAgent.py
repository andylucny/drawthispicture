import numpy as np
from agentspace import space, Agent
import time
import os

from clip import image_clip, text_clip, cosine_similarity, clip

def loadNames(fname):
    with open(fname,'rt',encoding='utf-8') as f:
        return [ line.strip() for line in f.readlines() ]

class NamingAgent(Agent):

    def __init__(self, nameFeatures, nameFocused, clip_threshold=0.2, judgement_threshold=0.7):
        self.nameFeatures = nameFeatures
        self.nameFocused = nameFocused
        self.clip_threshold = clip_threshold
        self.judgement_threshold = judgement_threshold*10
        super().__init__()
    
    def init(self):
        self.sk = loadNames('sk.txt')
        self.en = loadNames('en.txt')
        wipeout_path = 'wipeout.npy'
        if os.path.exists(wipeout_path):
            print('loading wipeout, please wait')
            self.wipeout = np.load(wipeout_path)
            print('wipeout loaded')
        else:
            print('calculating wipeout, please wait')
            self.wipeout = text_clip(self.en)
            np.save(wipeout_path,self.wipeout)
            print('wipeout ready')
        self.judgement = {} # index : -10..10
        self.last_index = -1
        space.attach_trigger(self.nameFeatures,self)

    def speak(self, text):
        space(validity=0.1)['tospeak'] = text

    def dontLook(self):
        space['dontLook'] = True

    def lookAround(self):
        space['dontLook'] = False

    def senseSelectAct(self):
        query = space[self.nameFeatures]
        if query is None:
            return
        
        probabilities = cosine_similarity(query, self.wipeout)
        index = np.argmax(probabilities)
        
        if probabilities[index] < self.clip_threshold:
            index = -1

        if index != -1:
            print(self.en[index],f'{probabilities[index]:.3f}')
        
        if index != -1 and index not in self.judgement:
            self.judgement[index] = 0

        remove_indices = []
        for i in self.judgement:
            self.judgement[i] = self.judgement[i] * 0.9 + 1 if i == index else -1
            if self.judgement[i] <= -self.judgement_threshold:
                remove_indices.append(i)
                
        for i in remove_indices:
            del self.judgement[i]
        
        if len(self.judgement) == 0:
            self.last_index = -1
            
        if index != -1 and index in self.judgement:
            if self.judgement[index] > self.judgement_threshold:
                if self.last_index != index:
                    text = f'Toto je asi {self.sk[index]}'
                    self.speak(text)
                    time.sleep(10)
