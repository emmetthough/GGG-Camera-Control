# Pixel Data Class
# v2.0, June 25, 2020
import numpy as np
from numpy.fft import rfft
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
        """
        Plots argmax bead position approximation over the dataset
        """
        times = np.arange(self.num_frames)
        x_means, y_means = self.bead_positions
        plt.plot(times, x_means, label='x')
        plt.plot(times, y_means, label='y')
        plt.xlabel('Time [frame_num]'); plt.ylabel('Pixel')
        plt.legend()
        plt.title('x and y positions of bead', fontsize=20)
        plt.show()

    def bead_temporal_fft(self, plot=False):
        """
        Plots the temporal fft for the bead positions, as given by self.bead_positions
        Precondition: track_mean has been called
        """
        try:
            freqs = np.linspace(0,int(self.num_frames/2),(int(self.num_frames/2))+1) 
            x_means, y_means = self.bead_positions

            x_fft = rfft(x_means)
            x_psd = (x_fft * x_fft.conj()).real
            y_fft = rfft(y_means)
            y_psd = (y_fft * y_fft.conj()).real

            if plot:
                plt.loglog(freqs,x_psd, label='x')
                plt.loglog(freqs,y_psd, label='y')
                plt.xlabel('Frequency [units??]'); plt.ylabel('Amplitude [arb.]')
                # plt.ylim(0, 1.5*max_y)
                plt.legend()
                plt.title('Temporal FFT of means', fontsize=20)
                plt.show()
            return freqs,x_psd,y_psd
        except:
            print('No bead position data! Call "track_mean" submethod first.')
            return        
