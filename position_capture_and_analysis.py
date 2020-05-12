# Bead position tracking and analysis script
# Emmett Hough, April 2020

import cv2
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.mlab import psd
from time import sleep
from pymba import Vimba
from typing import Optional
from pymba import Frame
from tqdm import tqdm
import shutil
from pixel_data import Pixel_Data

imageDir = r"C:\Users\Beads\Documents\EmmettH\data"
ROI_size = 100
MAX_HEIGHT = 480
MAX_WIDTH = 640

def startup():
    # Runs initial greeting and initializes trial params
    # Returns float frame rate, int duration, bool roi, string trial directory name
    
    # print("OpenCV version: " + cv2.__version__)
    # print()
    print("Position Analysis v210 \nEmmett Hough, Gratta Gravity Group")
    print()
    os.chdir("data")
    print("Current directory: ", os.getcwd())
    trialDir = input("Trial directory name: ")
    if trialDir != 'load':
        os.mkdir(os.path.join(imageDir, trialDir))
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
    # Takes a list of paths to images and returns a dictionary of the following structure:
    # key: "image name", value: numpy array
    images = []
    for img_path in image_path_list:
        print(img_path.split('_')[2])
        img = np.load(img_path)
        images.append(img)
    print()
    return images

def extract_mean(image_dict, plt_show=False):
    # Given frames in image_dict, returns position dict with the x,y position of the max of the mean 
    # i.e. extracts bead position and returns dict
    position_dict = {}
    for image_name in image_dict.keys():
        images = image_dict[image_name]
        # print("Horizontal size: ", len(images[0][0]))
        # print("Vertical size: ", len(images[0]))

        x_means = np.mean(images, axis=0)
        y_means = np.mean(images, axis=1)

        x_pos = np.argmax(x_means)
        y_pos = np.argmax(y_means)
        # run_avg = np.dot(means / range) / np.sum(range)

        position_dict[image_name] = (x_pos, y_pos)
    
    if plt_show:
        plt.plot(range(len(y_means)), y_means, label=image_name + ": y")
        plt.plot(range(len(x_means)), x_means, label=image_name + ": x")
        plt.legend()
        plt.show()

    return position_dict

def track_position(position_dict, plt_show=False):
    # This function assumes position_dict is for one video file
    # Tracks mean over time... bead tracking? FFT for all pixels found below
    x = []
    y = []
    for image in position_dict:
        x.append(position_dict[image][0])
        y.append(position_dict[image][1])
    plt.plot(range(len(x)), x, label="x-position vs time")
    plt.plot(range(len(y)), y, label="y-position vs time")
    plt.title("x and y positions over time")
    plt.xlabel("Time (frame)")
    plt.ylabel("Position (pixel)")
    plt.legend()
    if plt_show:
        plt.show()

# BELOW TO BE REPLACED WITH PIXEL DATA CLASS

def load_pixel_dict(images):
    # return: dict with -> key: (x,y) pixel identifier, value: [] list of values for each frame
    pixel_dict = {}
    keys = [key for key in images.keys()]
    test_img = images[keys[0]]
    image_height, image_width = np.shape(test_img)
    print('Image height: ', image_height)
    for x in range(image_width - 1):
        for y in range(image_height - 1):
            pixel_dict[(x,y)] = []
    print("Loading data...")
    for name in tqdm(images):
        frame = images[name]
        for x in range(ROI_size - 1):
            for y in range(ROI_size - 1):
                pixel_val = frame[x][y]
                pixel_dict[(x,y)].append(pixel_val)
    
    return pixel_dict

def load_psd_dict(pixel_dict, frame_rate):
    psd_dict = {}
    print()
    print("Loading PSDs...")
    for pixel in tqdm(pixel_dict.keys()):
        pixel_data = pixel_dict[pixel]
        (psd_data, freqs) = psd(pixel_data, Fs=frame_rate, detrend=mlab.detrend_none, NFFT=2**12)
        psd_dict[pixel] = (psd_data, freqs)
    
    return psd_dict

def psd_max(max_tuple, psd_dict):
    """
    max_pixel_list = []
    for (frame,pixel) in position_dict.items():
        if pixel not in max_pixel_list:
            max_pixel_list.append(pixel)
    for pixel in max_pixel_list:
        (psd, freqs) = psd_dict[pixel]
        plt.plot(freqs, psd, label=str(pixel))
    """
    (psd, freqs) = psd_dict[max_tuple]
    plt.plot(freqs, psd, label=str(pixel))
    plt.legend()
    plt.xscale('log')
    plt.title('Max Pixel PSD')
    plt.show()


def data_analysis(image_list):

    data = Pixel_Data(image_list)
    data.track_mean()
    data.plot_mean()
    data.bead_temporal_fft(plot=True)




def main(delete=True):
    set_camera_defaults()
    trial_params = startup()
    trial_num = 0
    trialDir = "trial_" + str(trial_num)
    while True:
        if trial_params[3] == 'load':
            trialDir = input('Load data directory: ')
            os.chdir(trialDir)
            delete = False
        else:
            os.mkdir(trialDir)
            os.chdir(trialDir)
            if trial_params[2]:
                set_roi()
            aquire_frames(trial_params[0], trial_params[1])
        images = load_images(create_image_path())
        data_analysis(images)
        # positions = extract_mean(images, plt_show=False)
        # psds = load_psd_dict(load_pixel_dict(images), trial_params[0])
        # psd_max(center, psds)
        # track_position(positions, plt_show=False)
        set_camera_defaults()
        os.chdir("..")

        new_trial = get_bool(input("Trial complete. Another? (Y/N): "))
        if delete:
            shutil.rmtree(trialDir)
        if new_trial:
            trial_num += 1
            trialDir = "trial_" + str(trial_num)
        else:
            print("Sounds good chief, have a super-duper day!")
            break


if __name__ == '__main__':
    main()
