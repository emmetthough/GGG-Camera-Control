# Pixel Data Class
# v2.0, June 25, 2020
import numpy as np
from numpy.fft import rfft
from numpy.fft import rfftfreq
from scipy.optimize import curve_fit
from skimage.registration import phase_cross_correlation
import matplotlib.pyplot as plt
import cv2

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
        frame = self.image_list[frame_num]
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
    
    def track_mean(self, roi_center=None, roi_size=None):
        """
        param roi_center: tuple (x,y) of center of roi
        param roi_size: int size of roi box
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
        
        if roi_center is not None and roi_size is not None:
            for i, image in enumerate(self.image_list):
                x_means[i] = roi_center[0] + (x_means[i] - roi_size/2)
                y_means[i] = roi_center[1] + (y_means[i] - roi_size/2)

        self.bead_positions = (x_means, y_means)

        return (x_means, y_means)

    def center_of_mass(self, roi_center=None, roi_size=None):

        x_com = []
        y_com = []
        for image in self.image_list:
            x_temp = []
            y_temp = []
            m_total = np.sum(image)
            for x in range(image.shape[1]):
                for y in range(image.shape[0]):
                    x_temp.append(image[y,x]*x)
                    y_temp.append(image[y,x]*y)
            x_com.append(int(np.sum(x_temp)/m_total))
            y_com.append(int(np.sum(y_temp)/m_total))
        
        if roi_center is not None and roi_size is not None:
            for i, image in enumerate(self.image_list):
                x_com[i] = roi_center[0] + (x_com[i] - roi_size/2)
                y_com[i] = roi_center[1] + (y_com[i] - roi_size/2)

        self.bead_positions = (x_com, y_com)
        
        return x_com, y_com

    def cv_center_of_mass(self, threshold, roi_center=None, roi_size=None):

        x_com = []
        y_com = []
        for img in self.image_list:
            ret,thresh = cv2.threshold(img,threshold,255,0)
            M = cv2.moments(thresh)
            x_com.append(int(M["m10"] / M["m00"]))
            y_com.append(int(M["m01"] / M["m00"]))
        
        if roi_center is not None and roi_size is not None:
            for i, image in enumerate(self.image_list):
                x_com[i] = roi_center[0] + (x_com[i] - roi_size/2)
                y_com[i] = roi_center[1] + (y_com[i] - roi_size/2)
        
        self.bead_positions = (x_com, y_com)

        return x_com,y_com

    def gaussian(self, x_lst, mu, sigma, amplitude, offset):
        norm = 1/(0.5*sigma*np.sqrt(2*np.pi))
        return amplitude*norm*np.exp(-(np.subtract(x_lst,mu))**2/(2*sigma**2))+offset

    def gaussian_fit(self, roi_center=None, roi_size=None, plot=False):
        
        i = 0
        images = self.image_list
        x_plot_params = []
        y_plot_params = []
        x_means = []
        y_means = []
        
        x_mu0 = np.argmax(np.mean(images[0], axis=0))
        y_mu0 = np.argmax(np.mean(images[0], axis=1))
        x_bounds=([x_mu0-5, -np.inf, -np.inf, -np.inf],[x_mu0+5, np.inf, np.inf, np.inf])
        y_bounds=([y_mu0-5, -np.inf, -np.inf, -np.inf],[y_mu0+5, np.inf, np.inf, np.inf])
        
        for img in images:
            print('Fit progress: {:2.1%}'.format(i / len(images)), end="\r")
            x_pixels = np.arange(img.shape[1])
            y_pixels = np.arange(img.shape[0])
            x_data = np.mean(img, axis=0)
            y_data = np.mean(img, axis=1)

            x_popt, x_pcov = curve_fit(self.gaussian, x_pixels, x_data, bounds=x_bounds, maxfev=2000)
            y_popt, y_pcov = curve_fit(self.gaussian, y_pixels, y_data, bounds=y_bounds, maxfev=2000)

            if plot and i == 0:
                x_plot_params.append(x_popt)
                x_plot_params.append(x_pixels)
                y_plot_params.append(y_popt)
                y_plot_params.append(y_pixels)
            i += 1
            
            x_means.append(x_popt[0])
            y_means.append(y_popt[0])

        if roi_center is not None and roi_size is not None:
            for i, image in enumerate(self.image_list):
                x_means[i] = roi_center[0] + (x_means[i] - roi_size/2)
                y_means[i] = roi_center[1] + (y_means[i] - roi_size/2)

        print()    
        if plot:
            plt.imshow(images[0])
            plt.plot(x_plot_params[1], self.gaussian(x_plot_params[1], *x_plot_params[0]), label='x')
            plt.plot(self.gaussian(y_plot_params[1], *y_plot_params[0]), y_plot_params[1], label='y')
            plt.scatter(x_means[0], y_means[0])
            plt.legend()
            plt.show()

        self.bead_positions = (x_means, y_means)
        
        return x_means, y_means

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

    def fft_norm(self, N, sampling_rate):
        return np.sqrt(2/(N*sampling_rate))

    def bead_temporal_fft(self, input_tuple=None, frame_rate=2000, plot=False):
        try:
            if input_tuple is None:
                x_means, y_means = self.bead_positions
            else:
                x_means, y_means = input_tuple

            freqs = rfftfreq(len(x_means), d=1./frame_rate)
            norm = self.fft_norm(len(x_means), frame_rate)
            x_fft = rfft(x_means)
            x_psd = np.sqrt(norm**2 * (x_fft * x_fft.conj()).real)
            y_fft = rfft(y_means)
            y_psd = np.sqrt(norm**2 * (y_fft * y_fft.conj()).real)

            if plot:
                plt.loglog(freqs,x_psd, label='x')
                plt.loglog(freqs,y_psd, label='y')
                plt.xlabel('Frequency [Hz]'); plt.ylabel('Amplitude [arb.]')
                # plt.ylim(0, 1.5*max_y)
                plt.legend()
                plt.title('Temporal FFT of means', fontsize=20)
                plt.show()
        except Exception as e:
            print('No bead position data! Call "track_mean" submethod first.')
            print(e)
            return
        return freqs,x_psd,y_psd

    def compare_signals(self, input1, input2):

        assert len(input1) == len(input2), 'Input lists different lengths'
        diff = np.abs(np.subtract(input1, input2))
        ratio = np.divide(input2, input1)
        return diff, ratio

    def single_pixel_intensity(self, position, frame_rate):
        
        vals = self.return_pixel_list(position)
        freqs, psd, _ = self.bead_temporal_fft(input_tuple=(vals,vals), frame_rate=frame_rate)
        return vals, freqs, psd
    
    def intensity(self, frame_rate=2000):
        vals = []
        for img in self.image_list:
            vals.append(np.mean(img))

        freqs = rfftfreq(len(vals), d=1./frame_rate)
        norm = self.fft_norm(len(vals), frame_rate)
        x_fft = rfft(vals)
        vpsd = np.sqrt(norm**2 * (x_fft * x_fft.conj()).real)
        
        return vals, freqs, vpsd

    def phase_correlation(self, images=None):

        if images is None:
            images = self.image_list

        xshifts = []
        yshifts = []
        for i,image in enumerate(images[1:]):
            print('Phase progress: {:2.1%}'.format(i / len(images)), end="\r")
            shift, error, diffphase = phase_cross_correlation(images[0], image, upsample_factor=100)
            xshifts.append(shift[0])
            yshifts.append(shift[1])
        # Add one to keep shape
        xshifts.append(xshifts[-1])
        yshifts.append(yshifts[-1])
        self.bead_positions = (xshifts, yshifts)
        return xshifts, yshifts
