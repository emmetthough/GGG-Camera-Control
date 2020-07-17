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

import time
from time import sleep
from pymba import Vimba
from typing import Optional
from pymba import Frame
from tqdm import tqdm

from pixel_data import Pixel_Data
#sys.path.append('/home/analysis_user/New_trap_code/Tools/')
import h5py
import BeadDataFile

imageDir = r"home/emmetth/EmmettH/data/"
ROI_size = 16
MAX_HEIGHT = 480
MAX_WIDTH = 640
MIN_EXPOSURE = 44.209

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
        camera.feature('ExposureTime').value = MIN_EXPOSURE
        camera.feature('AcquisitionFrameRateMode').value = 'Basic'


        camera.close()
    

def aquire_frames(frame_rate, duration, path):
    # Handles frame capture, according to params frame_rate and duration
    if os.getcwd() != path:
        os.chdir(path)
    
    global total # for use with progress bar in save_frame
    total = frame_rate*duration

    with Vimba() as vimba:
        vimba.startup()
        camera = vimba.camera(0)
        camera.open()

        # Create frame buffer queue
        buffer = 50
        frame_pool = [camera.new_frame() for _ in range(buffer)]
        for frame in frame_pool:
            # Tell camera about each frame in queue with designated callback
            frame.announce()
            frame.queue_for_capture(frame_callback=save_frame)

        # Initialize camera in MultiFrame mode and announce how many frames it will capture
        camera.feature('AcquisitionMode').value = 'MultiFrame'
        camera.feature('AcquisitionFrameCount').value = frame_rate*duration
        camera.feature('AcquisitionFrameRate').value = frame_rate 
        print("Starting Acquisition\n")

        # Start/stop acquisition
        start = time.time()
        camera.start_capture()
        camera.AcquisitionStart()

        sleep(duration)

        camera.AcquisitionStop()
        end = time.time()
        sleep(0.2)

        camera.end_capture()
        camera.flush_capture_queue()
        camera.close()

        vimba.shutdown()
        print('\n')
        print('total time: ', end-start)

    print('Capture complete\n')

    
def save_frame(frame: Frame):
    # Callable for camera.arm(), saves frame as numpy file
    global total
    frame_num = 'frame_{}'.format(frame.data.frameID)
    print("\rProgress: {:2.1%}".format(frame.data.frameID/total), end='\r')

    image = frame.buffer_data_numpy()
    np.save(frame_num, image)
    frame.queue_for_capture(frame_callback=save_frame)
        

def bead_height(imshow=False):
    # Same algorithm used with ROI set, but called seperately with option to show image
    # with highlighted bead
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

        camera.close()
        vimba.shutdown()

    if imshow:
        plt.figure()
        plt.imshow(image)
        plt.annotate('detected bead: {},{}'.format(x_pos,y_pos), xy=(x_pos,y_pos), xytext=(x_pos+25,y_pos+25), c='w')
        plt.scatter(x_pos, y_pos, c='r')
        plt.show()
    return x_pos, y_pos

        
def set_roi(size=ROI_size):
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

        # Grab brightest pixel from single image
        x_mean = np.mean(image, axis=0)
        y_mean = np.mean(image, axis=1)
        x_pos = int(np.argmax(x_mean))
        y_pos = int(np.argmax(y_mean))

        camera.feature('Height').value = size // 2 * 2 # increment of 2
        camera.feature('Width').value = size // 8 * 8 # must be increment of 8

        print("Center: ", x_pos, ",", y_pos)

        x_offset = 8 * ((x_pos - (size/2))//8)
        y_offset = 2 * ((y_pos - (size/2))//2)

        # In bounds
        if x_offset > 0 and x_offset + size < camera.feature('WidthMax').value:
            if y_offset > 0 and y_offset + size < camera.feature('HeightMax').value:
                camera.feature('OffsetX').value = int(x_offset)
                camera.feature('OffsetY').value = int(y_offset)
                print('Center ROI set')
        # x_offset too large
        elif x_offset + size > camera.feature('WidthMax').value:
            if y_offset + size > camera.feature('HeightMax').value:
                camera.feature('OffsetX').value = camera.feature('WidthMax').value - size
                camera.feature('OffsetY').value = camera.featrue('HeightMax').value - size
            else:
                camera.feature('OffsetX').value = camera.feature('WidthMax').value - size
                camera.feature('OffsetY').value = int(y_offset)
            print('x bound hit, ROI set')
        # y_offset too large
        elif y_offset + size > camera.feature('HeightMax').value:
            # bottom corner case covered in above elif
            camera.feature('OffsetX').value = int(x_offset)
            camera.feature('OffsetY').value = camera.featrue('HeightMax').value - size
            print('y bound hit, ROI set')
        # just set ROI in corner because I'm lazy
        else:
            camera.feature('OffsetX').value = 0
            camera.feature('OffsetY').value = 0
            print('Corner ROI set')

        camera.close()
        vimba.shutdown()

    return (x_pos,y_pos)

        
def create_image_path():
    #image path and valid extensions
    
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".bmp", ".npy", '.h5']
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
    return data.track_mean(), data.bead_temporal_fft()