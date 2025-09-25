from agentspace import Agent, space
import time
import re
from cloud import classify, generate
from drawingExtraction import extract_trajectories, visualize_trajectories
import cv2

class ResponderAgent(Agent):

    def __init__(self, nameText, namePicture, nameTrajectories):
        self.nameText = nameText
        self.namePicture = namePicture
        self.nameTrajectories = nameTrajectories
        super().__init__()
        
    def init(self):
        space.attach_trigger(self.nameText,self)
        
    def speak(self, text):
        space(validity=0.1)['tospeak'] = text
    
    def match(self,pattern,text):
        search = re.search(pattern,text)
        if search is None:
            self.groups = []
            return False
        else:
            self.groups = search.groups()
            return True
    
    def matched(self):
        return self.groups
        
    def senseSelectAct(self):
        text = space(default='')[self.nameText]
        if self.match(r'(O|o)pa( |r|)ku[a-zA-Z!]+ (.*)',text) or self.match(r'(R|r)e( |)peat (.*)',text):
            tobesaid = self.matched()[2]
            self.speak(tobesaid)
        elif self.match(r'(U|u)sm.* sa.*',text) or self.match(r'(S|s)mej.* sa.*',text) or self.match(r'(S|s)mile.*',text):
            print('happiness')
            space(validity=1.5)['emotion'] = "happiness"
        elif self.match(r'.*Ďakujem.*',text):
            self.speak('prosím')
        elif len(text) > 7: # to avoid void requests
            # call cloud
            print('question:',text)
            features = classify(text)
            if features is None:
                return
            kind = features[0]
            if kind == "image":
                image_prompt = features[1]
                caption = features[2]
                size = features[3]
                if caption is not None:
                    print('answer:',caption,'moment...')
                    self.speak(caption)
                if image_prompt is not None:
                    self.speak("... moment ...")
                    img = generate(image_prompt, size)
                    cv2.imwrite('generated.png',img)
                    trajectories = extract_trajectories(img)
                    if len(trajectories) > 0:
                        space[self.namePicture] = img
                        space[self.nameTrajectories] = trajectories
                        space['dontLook'] = False
                        print('we have trajectories')
                        result = visualize_trajectories(trajectories, img.shape)
                        cv2.imshow('picture',result)
                        cv2.waitKey(1)
            elif kind == "text":
                content = features[1]
                if content:
                    print('answer:',content)
                    self.speak(content)

if __name__ == '__main__':
    from SpeakerAgent import SpeakerAgent
    SpeakerAgent('tospeak')
    time.sleep(1)
    ResponderAgent('text')
    time.sleep(2)
    space['text'] = 'Opa kuj! toto mám povedať'
