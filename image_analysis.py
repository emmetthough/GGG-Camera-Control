# Bead position tracking and analysis script
# Emmett Hough, June 2020

import cv2
import os, os.path
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import psd

from time import sleep
from pymba import Vimba
from typing import Optional
from pymba import Frame
from tqdm import tqdm

from pixel_data import Pixel_Data
sys.path.append('/home/analysis_user/New_trap_code/Tools/')
import h5py
import BeadDataFile

imageDir = r"home/emmetth/EmmettH/data/"
ROI_size = 100
MAX_HEIGHT = 480
MAX_WIDTH = 640

def startup():
    # Runs initial greeting and initializes trial params
    # Returns float frame rate, int duration, bool roi, string trial directory name
    
    print("Position Analysis v3.0 \nEmmett Hough, Gratta Gravity Group")
    print()
    os.chdir("data")
    print("Current directory: ", os.getcwd())
    trialDir = input("Trial directory name ('load' for analysis): ")
    if trialDir != 'load':
        os.mkdir(trialDir)
        os.chdir(trialDir)
        frame_rate = float(input('Frame Rate: '))
        duration = int(input("Duration: "))
        print()
        roi = get_bool(input("Set ROI? (Y/N): "))
    else:
        frame_rate = 1
        duration = 1
        roi = False

    return [frame_rate, duration, roi, trialDir]


def set_camera_defaults():
    # Resets frame from ROI to default (full frame)
    with Vimba() as vimba:
        vimba.startup()
        camera = vimba.camera(0)
        camera.open()
        
        camera.feature('OffsetX').value = 0
        camera.feature('OffsetY').value = 0
        camera.feature('Height').value = MAX_HEIGHT
        camera.feature('Width').value = MAX_WIDTH

        camera.close()

        
def get_bool(input):
    # Helper function, intuitive
    if input[0].lower() == 'y':
        return True
    else:
        return False

    
def save_frame(frame: Frame):
    # Callable for camera.arm(), saves frame as numpy file
    frame_num = 'frame_{}'.format(frame.data.frameID)
    print(frame_num)
    image = frame.buffer_data_numpy()
    np.save(frame_num, image)
    

def aquire_frames(frame_rate, duration):
    # Handles frame capture, according to params frame_rate and duration
    with Vimba() as vimba:
        vimba.startup()
        camera = vimba.camera(0)
        camera.open()

        camera.feature('AcquisitionFrameRateMode').value = 'Basic'
        camera.feature('AcquisitionFrameRate').value = frame_rate 
        camera.arm('Continuous', save_frame)
        print("Starting Acquisition\n")
        camera.start_frame_acquisition()

        sleep(duration)

        camera.stop_frame_acquisition()

        camera.disarm()
        camera.close()

        
def set_roi():
    # Handles region of interest selection
    # Finds center of bead via averaging one frame and argmax to pick pixel
    # Vimba only accepts certain values of offsets, as found through Vimba Viewer
    # Assumes full frame input (i.e. no ROI set before this funtion is called)
    # Might need additional support for bead in corner of frame
    
    print()
    print('Setting ROI...\n')
    with Vimba() as vimba:
        vimba.startup()
        camera = vimba.camera(0)
        camera.open()

        camera.arm('SingleFrame')
        frame = camera.acquire_frame()
        image = frame.buffer_data_numpy()
        camera.disarm()

        x_mean = np.mean(image, axis=0)
        y_mean = np.mean(image, axis=1)
        x_pos = int(np.argmax(x_mean))
        y_pos = int(np.argmax(y_mean))

        camera.feature('Height').value = ROI_size # increment of 2
        camera.feature('Width').value = ROI_size + 4 # must be increment of 8

        print("Center: ", x_pos, ",", y_pos)

        x_offset = 8 * ((x_pos - (ROI_size/2))//8)
        y_offset = 2 * ((y_pos - (ROI_size/2))//2)

        if x_offset > 0 and x_offset < camera.feature('WidthMax').value:
            if y_offset > 0 and y_offset < camera.feature('HeightMax').value:
                camera.feature('OffsetX').value = int(x_offset)
                camera.feature('OffsetY').value = int(y_offset)
                print("ROI set successfully\n")
        else:
            print("Offset(s) out of range. Support coming soon.")
        
        camera.close()

        
def create_image_path():
    #image path and valid extensions
    
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".npy"]
    valid_image_extensions = [item.lower() for item in valid_image_extensions]
    
    #create a list all files in directory and
    #append files with a vaild extention to image_path_list
    for file in os.listdir(os.getcwd()):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(os.getcwd(), file))
    
    return image_path_list


def load_images(image_path_list):
    # Takes a list of paths to images and returns a list of the sorted frame arrays
    
    images = []
    # sort the string filenames by int frame number
    image_path_list.sort(key=lambda f: int(f.split('/')[-1].split('.')[0].split('_')[-1]))
    for img_path in image_path_list:
        img = np.load(img_path)
        images.append(img)
    return images


def load_h5(filename):
    # Loads a .h5 dataset into x, y, and z components
    
    bd = BeadDataFile.BeadDataFile(fname) #h5.wrapper written by Nadav
    x = bd.x2 # x coordinate, invisble for us
    y = bd.y2 # y coordinate, that is what you want
    z = bd.z2 # z coordinate, also what you want
    
    return (x,y,z)


def data_analysis(image_list):
    # Any data analysis wanted goes in here
    # Pixel_Data instance created, any submethods called on that
    
    data = Pixel_Data(image_list)
    data.track_mean()
    data.plot_mean()
    data.bead_temporal_fft(plot=True)


def main(delete=True):
    # set delete to 'False' to keep data after aquisition
    
    trial_params = startup()
    trial_num = 0
    trialDir = "trial_" + str(trial_num)
    
    while True:
        if trial_params[3] == 'load': # Analysis for loading a dataset
            trialDir = input('Load data directory: ')
            os.chdir(trialDir)
            delete = False
            if get_bool(input('h5 file? (y/n)')) == 'h5':
                fname = input('h5 filename:')
                x,y,z = load_h5(trialDir)
                print('h5 analysis coming in future version')
                complete = True
        else: # Complete analysis including data aquision 
            complete = False
            set_camera_defaults()
            os.mkdir(trialDir)
            os.chdir(trialDir)
            if trial_params[2]:
                set_roi()
            aquire_frames(trial_params[0], trial_params[1])
            
        if not complete: # Data analysis procedures as defined in data_analysis function
            images = load_images(create_image_path())
            data_analysis(images)

        os.chdir("..")

        if delete:
            shutil.rmtree(trialDir)

        if not trial_params[3] == 'load':   
            new_trial = get_bool(input("Trial complete. Another? (Y/N): "))
            if new_trial:
                trial_num += 1
                trialDir = "trial_" + str(trial_num)
            else:
                print("Sounds good chief, have a super-duper day!")
                break


if __name__ == '__main__':
    main()