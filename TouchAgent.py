from agentspace import Agent, space
import pyautogui
import pygame
import time
import ctypes
import numpy as np
import cv2

class TouchAgent(Agent):
            
    def init(self):
        # create a Pygame window
        pygame.font.init()
        global myfont
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        global screen
        screen = pygame.display.set_mode((2400, 1350), flags=pygame.NOFRAME, depth=0, display=1)
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        pygame.event.set_blocked(pygame.MOUSEBUTTONUP)
        pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN)
        pygame.display.set_caption('NICOs touchscreen')
        HWND = pygame.display.get_wm_info()['window']
        GWL_EXSTYLE = -20
        styles = ctypes.windll.user32.GetWindowLongA(HWND,GWL_EXSTYLE)
        WS_EX_NOACTIVATE = 0x08000000
        styles |= WS_EX_NOACTIVATE
        ctypes.windll.user32.SetWindowLongA(HWND,GWL_EXSTYLE,styles)
        screen_info = pygame.display.Info()
        width, height = screen_info.current_w, screen_info.current_h # 
        color_index = 0
        colors = [ (255,0,0), (0,255,0), (0,255,255), (80,80,255) ] # Red, Green, Cyan, Light Blue
        print('initialized')
        
        # Run the event loop
        #mouse = pyautogui.position()
        quit = False
        pygame.time.set_timer(pygame.USEREVENT + 1, 500)
        while not quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or self.stopped:
                    print('quit event')
                    quit = True
                elif event.type == pygame.FINGERDOWN:
                    # Process the first moment of touch
                    circle_color = colors[color_index]
                    color_index += 1
                    if color_index == len(colors):
                        color_index = 0
                    circle_radius = 3
                    circle_position = (int(width*event.x), int(height*event.y))
                    #print("touch detected at",circle_position)
                    pygame.draw.circle(screen, circle_color, circle_position, circle_radius)
                    pygame.display.flip()
                    space['touch'] = (event.x, event.y)
                elif event.type == pygame.FINGERMOTION:
                    circle_radius = 3
                    circle_position = (int(width*event.x), int(height*event.y))
                    #print("touch detected at",circle_position)
                    pygame.draw.circle(screen, circle_color, circle_position, circle_radius)
                    pygame.display.flip()                  
                    space['touch'] = (event.x, event.y)
                elif event.type == pygame.FINGERUP:
                    pass
                #elif event.type == pygame.KEYDOWN:
                #    if event.key == 1073741912: # Numeric ENTER
                #        print('stop key pressed')
                #        space['stop'] = True
                #    elif event.key ==  1073741920: # Numeric arrow up
                #        print('run key pressed')
                #        space['experiment'] = True
                #    #else:
                #   #    print('key',event.key)
                #else:
                #    mouse = pyautogui.position()
            pygame.display.flip()
            time.sleep(0.025)
            
        # quit Pygame
        print('quiting pygame')
        pygame.quit()

    def senseSelectAct(self):
        pass
    
if __name__ == "__main__":
   
    import os
    def quit():
        os._exit(0)

    TouchAgent()
    
