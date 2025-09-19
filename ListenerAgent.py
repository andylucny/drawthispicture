from agentspace import Agent, space
import numpy as np
import speech_recognition as sr
import torch

print("Available microphones:")
for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {mic_name}")

class ListenerAgent(Agent):

    def __init__(self, nameAudio, device_index=0):
        self.nameAudio = nameAudio
        self.device_index = device_index
        super().__init__()
        
    def init(self):
        r = sr.Recognizer()
        r.adjust_for_ambient_noise = True
        r.energy_threshold = 300
        r.pause_threshold = 0.3#0.8
        r.non_speaking_duration = 0.3
        r.dynamic_energy_threshold = False
        print('ready to listen')
        with sr.Microphone(device_index=self.device_index, sample_rate=16000) as source:
            while True:
                print('recording...')
                audio = r.listen(source)
                print('...recorded')
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                space(validity=2.0)[self.nameAudio] = torch_audio
 
    def senseSelectAct(self):
        pass

