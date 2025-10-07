from agentspace import Agent, space
import whisper

class TranscriptionAgent(Agent):

    def __init__(self, nameAudio, nameText):
        self.nameAudio = nameAudio
        self.nameText = nameText
        super().__init__()
        
    def init(self):
        print('loading whisper')
        self.audio_model = whisper.load_model("medium").to('cuda') # "base", "small", "medium", or "large" # large sa nevojde do 8GB
        print('ready to transcript')
        space.attach_trigger(self.nameAudio,self)
 
    def senseSelectAct(self):
        audio_data = space[self.nameAudio]
        if audio_data is not None:
            if len(audio_data) > 0:
                #print('transcripting')
                language = space(default='sk')['language']
                result = self.audio_model.transcribe(audio_data,language='slovak' if language == 'sk' else 'english')
                print('transcripted:',result['text'])
                space(validity=1.0)[self.nameText] = result['text']
                #print(result['text'])

if __name__ == '__main__':
    import time
    from ListenerAgent import ListenerAgent
    ListenerAgent('audio',1) #3 Jabra #1 ATR #1 IO2
    time.sleep(1)
    TranscriptionAgent('audio','text')
