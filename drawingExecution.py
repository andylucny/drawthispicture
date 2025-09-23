import os
import io
import requests
import zipfile
import numpy as np
import cv2
import time

from nicomover import setAngle, getAngle, enableTorque, disableTorque, park, release
from nicomover import current_posture, move_to_posture, load_movement, play_movement

method = 'perceptron' # 'rbf'

def download_zipfile(path,url):
    if os.path.exists(path):
        return
    print("downloading",url)
    response = requests.get(url)
    if response.ok:
        file_like_object = io.BytesIO(response.content)
        zipfile_object = zipfile.ZipFile(file_like_object)    
        zipfile_object.extractall(".")
    print("downloaded")
    
def download_nico_touch_model():
    #download_zipfile('nico-touch-right-arm.onnx','http://www.agentspace.org/download/nico-touch.zip')
    if not os.path.exists('nico-touch-right-arm.onnx'):
        shutil.copy('training/perceptron.onnx', 'nico-touch-right-arm.onnx')

if method == 'perceptron':
    import shutil
    import onnxruntime as ort
    download_nico_touch_model()
    dofs = ['r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x', 'r_indexfinger_x']
    providers = ['CPUExecutionProvider']
    touch_model = ort.InferenceSession('nico-touch-right-arm.onnx', providers=providers)
else:
    from scipy.interpolate import RBFInterpolator
    dataset = load_movement('dataset.txt')
    dofs = list(dataset[0].keys())[:10]
    touch_points = np.array([ [ sample[feat] for feat in ['x','y']] for sample in dataset], np.float32)
    touch_angles = np.array([ [ sample[feat] for feat in dofs] for sample in dataset], np.float32)
    interpolators = [ RBFInterpolator(touch_points, touch_angles[:, i]) for i in range(len(dofs)) ]

def interpolate_angles(input_points):
    interpolated = np.array([
        interpolator(input_points)
        for interpolator in interpolators
    ]).T
    return interpolated

def points2postures(points, resolution):
    inp = points / np.array([resolution],np.float32)
    if method == 'perceptron':
        out = touch_model.run(None, {"input": inp})[0]
    else:
        out = interpolate_angles(inp)
    postures = []
    for posture, inp_i in zip(out, inp):
        if method == 'perceptron':
            if posture[3] > 1.0: # elbow
                continue # not possible to reach
            if posture[5] > 1.0: # wrist-x
                posture[5] = 1.0
            posture = list(np.round(posture[:7]*180)) + [ round((inp_i[0]-0.5)*35), round((inp_i[1]-0.5)*5)-30 ]
        else:
            posture = list(np.round(posture)) + [ round((inp_i[0]-0.5)*35), round((inp_i[1]-0.5)*5)-30 ]
        postures.append(posture)
    return postures

def up(posture):
    posture_up = np.copy(posture)
    elbow = 3
    posture_up[elbow] -= 30.0 # [dg]
    return posture_up
    
def form_hand():
    dofs_ = ['r_thumb_z','r_thumb_x','r_indexfinger_x','r_middlefingers_x']
    posture = [-70.0, 26.0, -180.0, 172.0]
    move_to_posture(dict(zip(dofs_,posture)), speed=0.04, wait=False)

def get_ready():
    setAngle('r_elbow_y',55.0,speed=0.04)
    time.sleep(1)
    dofs_ = ['l_shoulder_z', 'l_shoulder_y', 'l_arm_x', 'l_elbow_y', 'l_wrist_z', 'l_wrist_x']
    posture = [-25.0, 9.0, 5.0, 96.0, -13.0, 53.0]
    move_to_posture(dict(zip(dofs_,posture)), speed=0.04, wait=False)
    dofs_ = ['r_shoulder_z', 'r_shoulder_y', 'r_arm_x', 'r_elbow_y', 'r_wrist_z', 'r_wrist_x', 'head_z', 'head_y']
    posture = [13.0, 24.0, 13.0, 93.0, 122.0, 180.0, 1.0, -40.0]
    move_to_posture(dict(zip(dofs_,posture)), speed=0.04, wait=True)

def move_arm(posture):
    dofs_ = dofs + [ 'head_z', 'head_y']
    move_to_posture(dict(zip(dofs_,posture)), speed=0.03, wait=True)

def scale_to_max_extent(actual_width, actual_height, max_width, max_height):
    # Calculate the scaling factors for both width and height
    width_ratio = max_width / actual_width
    height_ratio = max_height / actual_height
    # Choose the smaller ratio to maintain aspect ratio
    scale_factor = min(width_ratio, height_ratio)
    # Scale both dimensions
    new_width = int(actual_width * scale_factor)
    new_height = int(actual_height * scale_factor)
    return (new_width, new_height), scale_factor

def draw_trajectories(trajectories):
    all_points = [ point for trajectory in trajectories for point in trajectory ]
    rect = cv2.boundingRect(np.array(all_points))
    upper_border = 240
    resolution, scale_factor = scale_to_max_extent(rect[2], rect[3], 2400, 1350-upper_border)
    points_center = np.mean((all_points-np.array(rect[:2],np.float32))*scale_factor,axis=0)
    center = np.array((2400,1350),np.float32)/2
    offset = center - points_center
    enableTorque()
    form_hand()
    get_ready()
    sorted_trajectories = sorted(trajectories, key=len, reverse=True)
    for i, trajectory in enumerate(sorted_trajectories):
        if i == 0 or i == len(sorted_trajectories)-1:
            continue
        min_distance=1e9
        next_i = i
        for j in range(i,len(sorted_trajectories)):
            list1 = np.array(sorted_trajectories[i-1])
            list2 = np.array(sorted_trajectories[j])
            diffs = list1[:, np.newaxis, :] - list2[np.newaxis, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            distance = np.min(distances)
            if distance > min_distance:
                min_distance = distance
                next_i = j
        if next_i != i:
            sorted_trajectories[i], sorted_trajectories[next_i] = sorted_trajectories[next_i], sorted_trajectories[i]
        
    for trajectory in sorted_trajectories:
        points = (np.array(trajectory,np.float32) - np.array(rect[:2],np.float32)) * scale_factor
        points += offset
        postures = points2postures(points, (2400,1350))
        if len(postures) == 0:
            continue
        move_arm(up(postures[0]))
        time.sleep(0.7)
        for posture in postures:
            move_arm(posture)
        time.sleep(0.7)
        move_arm(up(postures[-1]))
        get_ready()
    
    park()

if __name__ == '__main__':
    def quit():
        os._exit(0)

    from nicomover import simulated
    if not simulated:
        from TouchAgent import TouchAgent
        TouchAgent()

    with open('picture.txt','rt') as f:
        trajectories = eval(f.read())
        
    draw_trajectories(trajectories)
