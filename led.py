import serial #pip install pyserial
import time
import numpy as np
import cv2 as cv

presets = [
    "happiness",
    "sadness",
    "anger",
    "disgust",
    "surprise",
    "fear",
    "neutral",
    "clear"
]

class LEDsimulator():

    def __init__(self, portname='COM3'):
        pass

    def decode(self, code):
        # Parameters for the grid
        rows, cols = 8, 16
        # Meaning of the grid
        halfbytes = np.array([
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32],
            [1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31],
            [1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31],
            [1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31],
            [1,3,5,7, 9,11,13,15,17,19,21,23,25,27,29,31],
        ])
        index = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],
            [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4],
        ])
        grid = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                grid[i,j] = int(code[halfbytes[i,j]-1],16) & (1 << index[i,j]-1)
        return grid
    
    def render(self, grid): # Function to draw the grid

        # Parameters for the grid
        rows, cols = 8, 16
        cell_size = 40  # Size of each cell in pixels
        grid_color = 255  # Color for white cells

        # Initialize a grid of zeros (black cells)
        img = np.zeros((rows*cell_size, cols*cell_size,3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                top_left = (j * cell_size, i * cell_size)
                bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
                color = (0,0,255) if grid[i, j] > 0 else (191,191,191) 
                cv.rectangle(img, top_left, bottom_right, color, -1)
                cv.rectangle(img, top_left, bottom_right, (0,0,0), 1)  # Grid lines
        
        return img
        
    def present(self,code):
        if code in presets:
            index = presets.index(code)
            presetcodes = [
                '000C1020404040404040404020100C00', # happiness
                '00201008080404040404040408182000', # sadness
                '00000078848444484848448484780000', # anger
                '002010081414141424242C3040408000', # disgust
                '00000000304884848484483000000000', # surprise
                '00000000304824242424483000000000', # fear
                '00000010301018181818181030100000', # neutral
                '00000000000000000000000000000000', # clear
            ]
            code = presetcodes[index]
        grid = self.decode(code)
        img = self.render(grid)
        # Function to update display
        cv.imshow("LED", img)
        cv.waitKey(1)

class LED():

    def __init__(self, portname='COM3'):
        # Initialize the serial connection
        try:
            print('trying to open',portname)
            self.ser = serial.Serial(
                port=portname,       # Port connected to the Arduino
                baudrate=9600,     # Baudrate for the communication
                bytesize=serial.EIGHTBITS,  # 8 data bits
                parity=serial.PARITY_NONE,  # No parity bit
                stopbits=serial.STOPBITS_ONE,  # 1 stop bit
                timeout=1          # Read timeout, adjust as needed
            )
            self.simulated = False
            print(f'LED on port {portname} opened')
            time.sleep(1)  # Give some time to establish the connection
        except serial.SerialException:
            print('LED simulated')
            self.simulated = True
            self.simulator = LEDsimulator()

    def send_preset(self, preset): # Function to send preset command
        if preset in presets:
            if self.simulated:
                self.simulator.present(preset)
            else:
                self.ser.write(preset.encode('utf-8'))  # Send the preset command
                time.sleep(0.1)
            #print(f'Sent preset: {preset}')
        else:
            print(f'Invalid preset: {preset}')

    def send_custom_bitmap(self, side, bitmap): # Function to send custom bitmap command
        if side not in ['l', 'r', 'm']:
            print(f'Invalid side: {side}')
            return

        if (len(bitmap) != 16 and side in ['l', 'r']) or (len(bitmap) != 32 and side == 'm'):
            print(f'Invalid bitmap length: {len(bitmap)}')
            return
        
        if self.simulated:
            if side in ['m']:
                self.simulator.present(bitmap)
        else:
            command = f"raw{side}{bitmap}"
            self.ser.write(command.encode('utf-8'))  # Send the custom bitmap command
            time.sleep(0.1)
        #print(f'Sent custom bitmap: {bitmap}')

    def close(self):
        # Close the serial connection
        if self.simulated:
            cv.destroyWindow('LED')
        else:
            self.ser.close()

if __name__ == '__main__':

    #sim = LEDsimulator()
    #sim.present('000C1020404040404040404020100C00')
    #time.sleep(1)
    #for _ in range(4):
    #    sim.present('neutral')
    #    time.sleep(0.25)
    #    sim.present('surprise')
    #    time.sleep(0.25)

    led = LED()

    # Send a preset command
    led.send_preset("happiness")
    time.sleep(2)

    # Wait and send a custom bitmap for left and right eyebrow and mouth (turn all LEDs on)
    led.send_custom_bitmap('l', 'FFFFFFFFFFFFFFFF')
    time.sleep(0.1)
    led.send_custom_bitmap('r', 'FFFFFFFFFFFFFFFF')
    time.sleep(0.1)
    led.send_custom_bitmap('m', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF')
    time.sleep(0.1)
    time.sleep(2)

    led.close()
    