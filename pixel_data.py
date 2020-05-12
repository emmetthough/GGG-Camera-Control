# Pixel Data Class
# v1.0, April 30, 2020
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

class Pixel_Data:
    def __init__(self, image_list):
        assert type(image_list) == list, 'Incorrect image_list type'
        assert type(image_list[0]) == np.ndarray, 'Frames not numpy arrays'
        self.image_list = image_list
        self.num_frames = len(image_list)
        self.frame_height = np.shape(image_list[0])[0]
        self.frame_width = np.shape(image_list[0])[1]

    def return_pixel_val(self, frame_num, position):
        """
        Params: 
        frame_num (int) 0-indexed number of frame of interest
        position (tuple) 0-indexed x-y position of pixel in question 
        Returns:
        pixel value of given frame at position (x,y)
        """
        x,y = position
        frame = image_list[frame_num]
        return frame[y,x]

    def return_pixel_list(self, position):
        """
        Params:
        position (tuple) 0-indexed x-y position of pixel in question
        Returns:
        List of that pixel's value at every frame in the trial
        """
        x,y = position
        pixel_vals = []
        for frame in self.image_list:
            pixel_vals.append(frame[y,x])

        return pixel_vals

    def track_pixels(self, x_list, y_list):
        """
        Params:
        x_list, y_list: list of corresponding x,y values for the pixel
        Returns:
        List of the pixel values corresponding to the x,y lists
        """
        assert len(x_list) == self.num_frames, 'x_list too short/long'
        assert len(y_list) == self.num_frames, 'y_list too short/long'
        pixel_vals = []
        for i in range(self.num_frames):
            x = x_list[i]
            y = y_list[i]
            frame = self.image_list[i]
            pixel_vals.append(frame[y,x])

        return pixel_vals
    
    def track_mean(self):
        """
        Returns tuple (x_list,y_list) containing the x,y position of the mean of 
        each frame in the trial (i.e. bead tracking)
        """
        x_means = []
        y_means = []

        for image in self.image_list:
            col_means = np.mean(image, axis=0)
            row_means = np.mean(image, axis=1)

            x_means.append(np.argmax(col_means))
            y_means.append(np.argmax(row_means))

        self.bead_positions = (x_means, y_means)

        return (x_means, y_means)

    def plot_mean(self):
        times = np.arange(self.num_frames)
        x_means, y_means = self.bead_positions
        plt.plot(times, x_means, label='x')
        plt.plot(times, y_means, label='y')
        plt.xlabel('Time [frame_num]'); plt.ylabel('Pixel')
        plt.legend()
        plt.title('x and y positions of bead', fontsize=20)
        plt.show()

    def bead_temporal_fft(self, plot=False):

        freqs = np.arange(self.num_frames)
        x_means, y_means = self.bead_positions

        x_fft = fft(x_means)
        y_fft = fft(y_means)

        print(np.max(x_fft))

        max_y = np.max([np.max(x_fft), np.max(y_fft)])

        if plot:
            plt.plot(freqs, x_fft, label='x')
            plt.plot(freqs, y_fft, label='y')
            plt.xscale('log')
            plt.xlabel('Frequency [units??]'); plt.ylabel('Amplitude [arb.]')
            plt.ylim(0, 1.5*max_y)
            plt.legend()
            plt.title('Temporal FFT of means', fontsize=20)
            plt.show()

        return x_fft,y_fft


            