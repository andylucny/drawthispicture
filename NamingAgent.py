import numpy as np
import cv2
from agentspace import space, Agent
import time
import os

from clip import image_clip, text_clip, cosine_similarity, clip
from nicomover import simulated
from sk import putText

def loadNames(fname):
    with open(fname,'rt',encoding='utf-8') as f:
        return [ line.strip() for line in f.readlines() ]

class NamingAgent(Agent):

    def __init__(self, nameFeatures, nameFocused, namePicture, clip_threshold=0.2, judgement_threshold=0.5):
        self.nameFeatures = nameFeatures
        self.nameFocused = nameFocused
        self.namePicture = namePicture
        self.clip_threshold = clip_threshold
        self.judgement_threshold = judgement_threshold*10
        self.drawing_announced = False
        super().__init__()
    
    def init(self):
        self.sk = loadNames('sk.txt')
        self.cz = loadNames('cz.txt')
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

        #if index != -1:
        #    print(self.en[index],f'{confidence:.3f} {self.judgement[index]:.2f}', space(default=False)[self.nameFocused])

        seeing_picture = False
        just_drawing = space['trajectories'] is not None
        if not just_drawing:
            self.drawing_announced = False
        if index != -1 and (self.en[index] == "Whiteboard" or self.en[index] == "Computer Box" or self.en[index] == 'Storage box'):
            picture = space[self.namePicture]
            if (not self.drawing_announced) and (not picture is None):
                picture_query = image_clip(picture)
                index, confidence = self.nameIt(picture_query)
                if index != -1:
                    seeing_picture = True
                    self.drawing_announced = True
                    print('picture',self.en[index],f'{confidence:.3f} {self.judgement[index]:.2f}', space(default=False)[self.nameFocused])

        image = space['robotEye']
        if image is not None:
            image = np.copy(image)
            y = 40
            for choice in self.judgement:
                score = self.judgement[choice]
                if score > 0:
                    color = (0,0,255) if score > self.judgement_threshold else (0,255,255)
                    lang = space['language']
                    if lang == 'sk':
                        putText(image,f'{self.sk[choice]} ... {score:.2f}',(10,y),0,1.0,color,2)
                    elif lang == 'cz':
                        putText(image,f'{self.cz[choice]} ... {score:.2f}',(10,y),0,1.0,color,2)
                    else:
                        cv2.putText(image,f'{self.en[choice]} ... {score:.2f}',(10,y),0,1.0,color,2)
                    y += 40
            cv2.imshow('Naming',image)
            cv2.waitKey(1)
        
        if index != -1 and index in self.judgement:
            if self.judgement[index] > self.judgement_threshold or seeing_picture:
                if (self.last_index != index or seeing_picture) and \
                    self.en[index] != 'Desk' and self.en[index] != 'Tablet' and self.en[index] != 'laptop' and self.en[index] != 'Apartment' and \
                    self.en[index] != 'Desktop' and self.en[index] != 'Computer' and self.en[index] != 'Game board' and \
                    self.en[index] != 'Monitor' and self.en[index] != 'Projector' and self.en[index] != 'Photographer' and \
                    self.en[index] != 'Laptop' and self.en[index] != 'Blackboard' and self.en[index] != 'Whiteboard' and \
                    self.en[index] != 'Computer Box' and self.en[index] != 'Storage box' and self.en[index] != 'Dinning Table' and \
                    self.en[index] != 'Coffee Table': # Desk and others are in the front of the robot
                    if simulated or space(default=False)[self.nameFocused] or seeing_picture:
                        print(self.en[index],f'{confidence:.3f} {self.judgement[index]:.2f}', space(default=False)[self.nameFocused])
                        lang = space(default='en')['language']
                        if lang == 'sk':
                            if self.sk[index] == 'deti':
                                text = f'Toto sú asi {self.sk[index]}.'
                            else:
                                text = f'Toto je asi {self.sk[index]}.'
                            if seeing_picture:
                                text += '. Nakreslíme to!'
                        elif lang == 'cz':
                            if self.cz[index] == 'deti':
                                text = f'Tohle budou {self.cz[index]}.'
                            else:
                                text = f'Tohle bude {self.cz[index]}.'
                            if seeing_picture:
                                text += '. Nakreslíme to!'
                        else:
                            text = f'Perhaps, this is a{"n" if self.en[index][0] in ["a","e","i","o","u"] else ""} {self.en[index]}.'
                            if seeing_picture:
                                text += ". Let's draw it!"
                        self.speak(text)
                        self.last_index = index
                        
                        for t in [5,54,543,5432,54321]:
                            if image is not None:
                                cv2.putText(image,f'taking rest {t}',(10,y),0,1.0,color,2)
                                cv2.imshow('Naming',image)
                                cv2.waitKey(1)
                            time.sleep(1)
