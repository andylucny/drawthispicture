import numpy as np
from agentspace import space, Agent
import time
import os

from clip import image_clip, text_clip, cosine_similarity, clip
from nicomover import simulated

def loadNames(fname):
    with open(fname,'rt',encoding='utf-8') as f:
        return [ line.strip() for line in f.readlines() ]

class NamingAgent(Agent):

    def __init__(self, nameFeatures, nameFocused, clip_threshold=0.2, judgement_threshold=0.5):
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
        
    def nameIt(self, query):
        top = 3
        probabilities = cosine_similarity(query, self.wipeout)
        indices_above_threshold = np.where(probabilities >= self.clip_threshold)[0]
        filtered_probabilities = probabilities[indices_above_threshold]
        top_filtered_indices = np.argsort(filtered_probabilities)[-top:][::-1]
        top_indices = indices_above_threshold[top_filtered_indices]
        
        for i in top_indices:
            if i not in self.judgement:
                self.judgement[i] = 0

        remove_indices = []
        for i in self.judgement:
            self.judgement[i] = self.judgement[i] * 0.9 + 1 if i in top_indices else -1
            if self.judgement[i] <= -self.judgement_threshold:
                remove_indices.append(i)
                
        for i in remove_indices:
            del self.judgement[i]
        
        promissing = {i:self.judgement[i] for i in top_indices if i in self.judgement}
        index = max(promissing, key=promissing.get, default=-1)
        return index, probabilities[index] if index != -1 else 0.0

    def senseSelectAct(self):
        query = space[self.nameFeatures]
        if query is None:
            return
        
        index, confidence = self.nameIt(query)

        if index != -1:
            print(self.en[index],f'{confidence:.3f} {self.judgement[index]:.2f}', space(default=False)[self.nameFocused])
        #else:
        #    self.last_index = -1

        seeing_picture = False
        if index != -1 and self.en[index] == "Whiteboard":
            picture = space['picture']
            if not picture is None:
                picture_query = image_clip(picture)
                index, confidence = self.nameIt(picture_query)
                if index != -1:
                    seeing_picture = True
                    print('picture',self.en[index],f'{confidence:.3f} {self.judgement[index]:.2f}', space(default=False)[self.nameFocused])

        if index != -1 and index in self.judgement:
            if self.judgement[index] > self.judgement_threshold or seeing_picture:
                if self.last_index != index and self.en[index] != 'Desk': # Desk is in the front of the robot
                    if simulated or space(default=False)[self.nameFocused] or seeing_picture:
                        if space(default='en')['language'] == 'sk':
                            text = f'Toto je asi {self.sk[index]}.'
                            if seeing_picture:
                                text += '. Nakreslíme to!'
                        else:
                            text = f'Perhaps, this is a{"n" if self.en[index][0] in ["a","e","i","o","u"] else ""} {self.en[index]}.'
                            if seeing_picture:
                                text += ". Let's draw it!"
                        self.speak(text)
                        self.last_index = index
                        time.sleep(8)
