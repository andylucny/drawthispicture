import pyttsx3
import os
from agentspace import space

space['language'] = 'sk'
    
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    if len(voices) == 0:
        import platform 
        if platform.system() == "Windows":
            print("run Registy Editor (regedit) and")
            print("export content of Computer\\HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens into my.reg file" )
            print("rewrite paths in file to Computer\\HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens")
            print("import the my.reg file")
        else:
            print("install:")
            print("$ sudo apt-get install espeak -y")
        os._exit(0)
        
    voice_names = [ voice.name for voice in voices ] 
    print(voice_names)
    
    language = space(default='en')['language']
    if language == 'sk':
        try:
            speaker = voice_names.index('Microsoft Filip - Slovak (Slovakia)')
        except ValueError:
            try:
                speaker = voice_names.index('Vocalizer Expressive Laura Harpo 22kHz')
            except ValueError:
                speaker = 0
                
    elif language == 'cz':
        try:
            speaker = voice_names.index('Microsoft Jakub - Czech (Czech Republic)')
        except ValueError:
            try:
                speaker = voice_names.index('Vocalizer Expressive Laura Harpo 22kHz')
            except ValueError:
                speaker = 0
    
    else:
        try:
            speaker = voice_names.index('Microsoft Zira Desktop - English (United States)')
        except ValueError:
            try:
                speaker = voice_names.index('Microsoft David Desktop - English (United States)')
            except ValueError:
                try:
                    speaker = voice_names.index('english-us')
                except ValueError:
                    speaker = 0
    
    #print('speaker:',speaker, voices[speaker].name)
    space['speaking'] = True
    engine.setProperty('voice', voices[speaker].id)
    engine.say(text)
    print('speaking on <'+text+'>')
    engine.runAndWait()
    print('speaking off')
    space['speaking'] = False

if __name__ == "__main__":
    speak('Na holi sa pasie ovca.')
    space['language'] = 'cz'
    speak('Tohle by mohla být ovce')
    print('done')
