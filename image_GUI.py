# Analysis GUI script
# Emmett Hough, Summer Research 2020
# Most recent update: August 27, 2020

import sys
import os
import traceback
from time import sleep
from datetime import datetime

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

import numpy as np
from numpy.fft import rfft
from numpy.fft import rfftfreq

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import camera_control_analysis as cca
from pixel_data import Pixel_Data
import BeadDataFile

LARGE_FONT = ("Verdana 24 bold")
MEDIUM_FONT = ("Verdana 18")

global ROI_size
global imageDir
global path 
path = os.getcwd()
global cam_id
cam_id = 2
global loaded_dict
loaded_dict = {}

class analysis_GUI(tk.Tk):
    # Main wrapper for program, initializes the four pages and tkinter controls
    def __init__(self):

        # Initialize
        tk.Tk.__init__(self)
        tk.Tk.wm_title(self, "Image Analysis")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Initialize different pages
        self.frames = {}
        for page in (StartPage, AcquisitionPage, AnalysisPage):

            frame = page(container, self)
            self.frames[page] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Show Start Page
        self.show_frame(StartPage)
    
    def show_frame(self, container):
        frame = self.frames[container]
        frame.tkraise()
    

class StartPage(tk.Frame):

    def __init__(self, parent, controller):

        # Initialize with navigation buttons
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Welcome to Image Analysis", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.frame1 = tk.Frame(self)
        button = tk.Button(self.frame1, text="Acquisition", command=lambda: controller.show_frame(AcquisitionPage))
        button.grid(row=0, column=0, padx=5, pady=5)

        button2 = tk.Button(self.frame1, text="Analysis", command=lambda: controller.show_frame(AnalysisPage))
        button2.grid(row=0, column=1, padx=5, pady=5)

        button3 = tk.Button(self.frame1, text="Comparison", command=lambda: self.raise_compare(parent, controller))
        button3.grid(row=0, column=2, padx=5, pady=5)

        self.frame1.pack()

    def raise_compare(self, parent, controller):
        # Compare page takes loaded_dict as argument, so needs separate launcher
        frame = ComparisonPage(parent, controller, None)
        frame.grid(row=0, column=0, sticky='nsew')
        frame.tkraise()
        controller.show_frame(AnalysisPage)


class AcquisitionPage(tk.Frame):
    # Controls camera frame data acquisition and data storage with directories
    def __init__(self, parent, controller):

        # Initialize frames for GUI elements
        tk.Frame.__init__(self, parent)

        frame1 = tk.Frame(self)
        frame1.pack(side=tk.TOP, fill=tk.X)

        frame2 = tk.Frame(self)
        frame2.pack(side=tk.TOP, fill=tk.X)

        frame3 = tk.Frame(self)
        frame3.pack(side=tk.LEFT, fill=tk.X, expand=tk.YES)
        
        frame4 = tk.Frame(self)
        frame4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=tk.YES)

        # Home button and title
        label = tk.Label(frame1, text="Acquisition", font=LARGE_FONT)
        label.pack(pady=5,padx=10)
        home = tk.Button(frame1, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        home.pack()
        change = tk.Button(frame1, text="Change working directory", command=self.change_path)
        change.pack()

        # Controls and Console Text
        controls = tk.Label(frame2, text="Controls", font=MEDIUM_FONT)
        console = tk.Label(frame2, text="Console Output", font=MEDIUM_FONT)

        controls.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        console.grid(row=0, column=1, sticky='ew', padx=5, pady=5)

        frame2.columnconfigure(0, weight=1)
        frame2.columnconfigure(1, weight=1)
        frame2.rowconfigure(0, weight=1)

        # Controls frame
        framerate_lbl = tk.Label(frame3, text="Frame Rate:")
        self.framerate_ent = tk.Entry(frame3, width=10)

        duration_lbl = tk.Label(frame3, text="Duration:")
        self.duration_ent = tk.Entry(frame3, width=10)

        trialpath_lbl = tk.Label(frame3, text="Trial name:")
        self.trialpath_ent = tk.Entry(frame3, width=10)

        self.roi_var = tk.BooleanVar()
        roi_chkbtn = tk.Checkbutton(frame3, text="ROI", variable=self.roi_var, onvalue=True, offvalue=False)

        ROIs = [16, 32, 48, 64, 80]
        self.roi_intvar = tk.IntVar(frame2)
        self.roi_intvar.set(ROIs[0])
        ROI_menu = tk.OptionMenu(frame3, self.roi_intvar, *ROIs)

        roi_selectext = tk.Label(frame3, text='ROI size:')
        
        self.show_var = tk.BooleanVar()
        show_chkbtn = tk.Checkbutton(frame3, text="Show analysis after", variable=self.show_var, onvalue=True, offvalue=False)

        self.imshow_var = tk.BooleanVar()
        imshow_chkbtn = tk.Checkbutton(frame3, text='Show Test Image', variable=self.imshow_var, onvalue=True, offvalue=False)

        height_btn = tk.Button(frame3, text="Bead Height", command=self.bead_height)

        start_btn = tk.Button(frame3, text="START", relief=tk.RAISED, command=lambda: self.start_acquisition(controller))

        # geometry manager for controls frame
        framerate_lbl.grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.framerate_ent.grid(row=0, column=1, sticky='e', padx=5, pady=5)
        duration_lbl.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.duration_ent.grid(row=1, column=1, sticky='e', padx=5, pady=5)
        trialpath_lbl.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.trialpath_ent.grid(row=2, column=1, stick='e', padx=5, pady=5)
        roi_chkbtn.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        roi_selectext.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        ROI_menu.grid(row=3, column=1, sticky='e', padx=5, pady=5)
        height_btn.grid(row=4, column=0, sticky='w', padx=5, pady=5)
        imshow_chkbtn.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        start_btn.grid(row=5, column=0, sticky='nsew', padx=5, pady=5)
        show_chkbtn.grid(row=5, column=1, sticky='w', padx=5, pady=5)

        # Make elements expand with window resize
        for col in range(2):
            frame3.columnconfigure(col, weight=1)
        for row in range(6):
            frame3.rowconfigure(row, weight=1)
        
        # Console frame
        scrollbar = tk.Scrollbar(frame4) 
        self.listbox = tk.Listbox(frame4, width=50)
        scrollbar.pack(side=tk.RIGHT, pady=5)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.insert(tk.END, path)
 
    def start_acquisition(self, controller):
        # Called upon push of START button, handles directory management and starts frame acquisition

        if os.getcwd() != path:
            os.chdir(path)
        try: 
            framerate = int(self.framerate_ent.get())
            duration = int(self.duration_ent.get()) 

            self.listbox.insert(tk.END, 'Starting Acquisition...')
            self.listbox.update_idletasks()

            imageDir = os.path.join(os.getcwd(), self.trialpath_ent.get())

            try: # New directory
                os.chdir(imageDir)
            except : # Existing directory
                os.mkdir(imageDir)
                os.chdir(imageDir)

            # add trial sub-directory in imageDir
            highest = 0
            if len(os.listdir(os.getcwd())) != 0:
                for file in os.listdir(os.getcwd()):
                    if file[0] == 't': # only trial_* subdirectories
                        num = int(file.split('_')[-1])
                        if num > highest:
                            highest = num

            highest += 1
            os.mkdir("trial_{}".format(highest))
            os.chdir("trial_{}".format(highest))
            self.trialnum = highest
            
            global cam_id
            cca.set_camera_defaults(id=cam_id)

            # Set ROI if checkbutton is true
            if self.roi_var.get():
                self.listbox.insert(tk.END, 'Setting ROI...')
                self.listbox.update_idletasks()

                global ROI_size
                ROI_size = self.roi_intvar.get()
                x, y = cca.set_roi(size=ROI_size, id=cam_id)
                global roi_center
                roi_center = (x,y)

                self.listbox.insert(tk.END, 'ROI ({} pixels wide) set: center at {}, {}'.format(ROI_size,x,y))
                self.listbox.update_idletasks()

            sleep(0.5)

            # Start frame acquisition
            self.listbox.insert(tk.END, 'Starting frame capture')
            self.listbox.update_idletasks()
            cca.aquire_frames(framerate, duration, os.getcwd(), id=cam_id)
            cca.set_camera_defaults(id=cam_id)

            # Check expected number of frames was captured
            num_frames = len(os.listdir(os.getcwd()))
            self.listbox.insert(tk.END, '{} frames successfully captured'.format(num_frames))
            self.listbox.insert(tk.END, '')
            self.listbox.update_idletasks()

            # Write metadata.txt to trial folder
            metadata = open('metadata.txt', 'w+')
            date_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            metadata.write('Date: {}\n'.format(date_time))
            metadata.write('Frame Rate: {}\n'.format(framerate))
            metadata.write('Duration: {}\n'.format(duration))
            metadata.write('ROI size: {}\n'.format(ROI_size))
            metadata.write('ROI center: {}\n'.format(roi_center))
            metadata.close()

            os.chdir('..')
            os.chdir('..')

            if self.show_var.get():
                # Pop up analysis window if designated
                frame = analysis_GUI().frames[AnalysisPage]
                frame.load_from_acquisition(path=imageDir)
                frame.tkraise()
            
        except Exception as e:
            self.listbox.insert(tk.END, 'Acquisition failure: {}'.format(e))
            traceback.print_exc()
          
    def change_path(self):
        # Called upon change path button, handles directory selection and change

        global path
        path = filedialog.askdirectory(initialdir=os.getcwd())
        os.chdir(path)
        self.listbox.delete(0, tk.END)
        self.listbox.insert(tk.END, os.getcwd())
        print('Acquisition path:', os.getcwd())

    def bead_height(self):
        # Get position of bead with argmax method, either print out or show image

        global cam_id
        cca.set_camera_defaults(id=cam_id)
        x,y = cca.bead_height(self.roi_intvar.get(), imshow=self.imshow_var.get(), id=cam_id)
        if not self.imshow_var.get():
            self.listbox.insert(tk.END, "Bead position: {}, {}".format(x,y))
            self.listbox.update_idletasks()


class AnalysisPage(tk.Frame):
    # Provides an initial analysis of bead positions, and stores analysis methods for fast access afterwards
    def __init__(self, parent, controller):

        # Initialize with buttons
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Analysis", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        frame1 = tk.Frame(self)
        frame1.pack()

        self.parent = parent
        self.controller = controller

        # Initialize GUI buttons
        home_btn = tk.Button(frame1, text="Back to HOME", command=lambda: controller.show_frame(StartPage))
        home_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        load_btn = tk.Button(frame1, text="LOAD DATA (from directory)", command=self.change_trialpath)
        load_btn.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        position_label = tk.Label(frame1, text='Position Algorithm:')
        position_label.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)

        self.position_type_var = tk.StringVar(frame1)
        position_types = ['argmax', 'cv_com', 'gaussian', 'phase correlation']
        self.position_type_var.set(position_types[0])
        position_menu = tk.OptionMenu(frame1, self.position_type_var, *position_types)
        position_menu.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
    
        self.imageDir = os.getcwd()

        self.imageDir_lbl = tk.Label(self, text='Current trial dir: ' + self.imageDir)
        self.imageDir_lbl.pack()

        # Initialize dict of loaded numpy files for faster access on a single session
        self.images = {}
        self.data_loaded = False

    def change_trialpath(self):
        # Called on load from directory button, handles selection and sets attribute for access across methods
        try:
            self.imageDir = filedialog.askdirectory(initialdir=os.getcwd())
            os.chdir(self.imageDir)
            self.imageDir_lbl.config(text='Current trial dir: ' + self.imageDir)
            self.imageDir_lbl.update_idletasks()

            self.populate_images()

        except Exception as e:
            print('change_trialpath() ERROR: ', e)

    def load_from_acquisition(self, path=os.getcwd()):
        # Called from Acquisition frame to pop up and populate analysis frame

        try:
            self.imageDir = path
            self.imageDir_lbl.config(text='Current trial dir: ' + self.imageDir)
            self.imageDir_lbl.update_idletasks()

            os.chdir(path)

            self.populate_images()
        except:
            print('No stored acquisition data... Manually select trial directory.')
    
    def populate_images(self):
        # Reads in (sorted) files from image directory, populates trial buttons

        trials = []
        for file in os.listdir(self.imageDir):
            if file[0] == 't':
                trials.append(file)
        trials.sort(key=lambda t: int(t.split('_')[-1])) # sort by trial_num
        
        global loaded_dict
        if self.imageDir not in loaded_dict.keys() or len(loaded_dict[self.imageDir].keys()) != len(trials):
            loaded_dict[self.imageDir] = {}
            # Populate and load dictionary with numpy arrays from each trial in directory
            for trialnum in trials:
                print('Loading {}:'.format(trialnum))
                os.chdir(trialnum)
                npfile_lst = cca.load_images(cca.create_image_path())
                os.chdir('..')
                self.images[trialnum] = npfile_lst
                loaded_dict[self.imageDir][trialnum] = npfile_lst
        else:
            self.images = loaded_dict[self.imageDir]

        print()
        print('Data loaded!')
        print()

        # Make and populate buttons for each trial in directory
        # Initialize data for each button, which pulls up graphs upon call
        trialframe = tk.Frame(self)
        col = 0
        for trial in trials:
            but = tk.Button(trialframe, text=trial, command=lambda trial=trial: self.analysis_graphs(self.images[trial], trial))
            but.grid(row=0, column=col, sticky='nsew', padx=5, pady=5)
            col += 1
        qpd_btn = tk.Button(trialframe, text="COMPARE", command=self.raise_compare)
        qpd_btn.grid(row=0, column=col, sticky='nsew', padx=5, pady=5)
        trialframe.pack()
    
    def single_pixel_intensity(self, images,frame_rate, N=500, random='gaussian', plot=True):
        # Randomly samples N points from guassian or normal distribution, uses that pixel's value
        # over the set of frames as input for a PSD, then averages all PSDs

        data = Pixel_Data(images)

        if random == 'gaussian':
            x = np.random.normal(loc=0.5, scale=0.2, size=N)*images[0].shape[1]
            y = np.random.normal(loc=0.5, scale=0.2, size=N)*images[0].shape[0]
        elif random == 'uniform':
            x = np.random.randint(0, images[0].shape[1], N)
            y = np.random.randint(0, images[0].shape[0], N)
        else:
            print('Invalid random param')
            return

        psds = None
        freqs = None
        for i in range(N):
            # Make sure sample is in range
            if x[i] >= images[0].shape[1] - 1:
                x[i] = images[0].shape[1] - 1
            if y[i] >= images[0].shape[0] - 1:
                y[i] = images[0].shape[0] - 1
            if x[i] < 0:
                x[i] = 0
            if y[i] < 0:
                y[i] = 0
        
            vals, freq, psd = data.single_pixel_intensity((int(round(x[i])),int(round(y[i]))), frame_rate)
            if i == 0:
                freqs = freq
                psds = psd
            else:
                psds = np.vstack((psds, psd))

        psd_avg = np.mean(psds, axis=0)
        
        if plot:
            plt.loglog(freqs, psd_avg)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [arb.]')
            plt.title('Single-Pixel Intensity Avg, {} {} samples'.format(N, random))
            plt.show()

        return freqs, psd_avg

    def analysis_graphs(self, npfile_lst, trialnum):
        # Called on trial button push, handles canvas and graphing

        if os.getcwd() != self.imageDir:
            os.chdir(self.imageDir)

        self.frame1 = tk.Frame(self)

         # Get data/metadata
        try:
            os.chdir(trialnum)
            files = os.listdir(os.getcwd())
            metadata_file = [f for f in files if f == 'metadata.txt'][0]
            metadata = open(metadata_file, 'r')
            for line in metadata:
                attr = line.split(':')
                if attr[0] == 'Frame Rate':
                    frame_rate = int(attr[1].strip())
                if attr[0] == 'Duration':
                    duration = attr[1].strip()
                if attr[0] == 'ROI size':
                    roi_size = int(attr[1].strip())
                if attr[0] == 'ROI center':
                    roi_center = self.grab_tuple(attr[1].strip())
            os.chdir('..')
        except:
            print('Error in metadata parsing')
            return
        
        single_pxl_btn = tk.Button(self.frame1, text='Single-Pixel Intensity', command=lambda: self.single_pixel_intensity(npfile_lst, frame_rate))
        single_pxl_btn.pack(padx=5, pady=5)

        self.frame1.pack()

        try: # Clear existing frames
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
        except: # On first call
            self.fig = Figure(figsize=(5,6), dpi=100)
            self.fig.subplotpars.hspace = 0.5

            self.ax1 = self.fig.add_subplot(311)
            self.ax2 = self.fig.add_subplot(312)
            self.ax3 = self.fig.add_subplot(313)

            self.vbar = tk.Scrollbar(self.frame1, orient=tk.VERTICAL)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame1)
            self.canvas.get_tk_widget().config(width=800, height=1000, scrollregion=(0,0,800,800))
            self.canvas.get_tk_widget().config(yscrollcommand=self.vbar.set)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

            self.vbar.pack(side=tk.RIGHT, fill=tk.Y, expand=1)
            self.vbar.config(command=self.canvas.get_tk_widget().yview)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()
            self.canvas.mpl_connect('key_press_event', self.on_key_press)


        os.chdir(trialnum) # Change into trial directory for saving analysis sets in cca.data_analysis function
        position_type = self.position_type_var.get()
        print('Using {} position algorithm'.format(position_type))
        means, ffts = cca.data_analysis(npfile_lst, mean=position_type, frame_rate=frame_rate, roi_center=roi_center, roi_size=roi_size)
        freqs, xfft, yfft = ffts
        vals, vfreqs, vpsd = Pixel_Data(npfile_lst).intensity(frame_rate=frame_rate)
        times = np.linspace(0, int(duration), len(means[0]))
        os.chdir('..')

        # Plot!
        title = trialnum + ': {} fps for {} sec with {} pixel-wide ROI'.format(frame_rate, duration, str(roi_size))
        self.fig.suptitle(title)
        self.ax1.plot(times, means[0], label='y')
        self.ax1.plot(times, means[1], label='z')
        self.ax2.plot(freqs, xfft, label='y')
        self.ax2.plot(freqs, yfft*100, label='z*100')
        self.ax3.plot(vfreqs, vpsd)
        self.ax1.set_title("Bead positions, using {} method".format(position_type))
        self.ax1.set(xlabel="Time [sec]", ylabel="Bead position [pixel]")
        self.ax2.set_title("Position PSDs")
        self.ax2.set(xlabel="Frequency [Hz]", ylabel="PSD [arb.]")
        self.ax3.set_title("Intensity PSD")
        self.ax3.set(xlabel="Frequency [Hz]", ylabel="PSD [arb.]")
        self.ax1.legend()
        self.ax2.loglog()
        self.ax2.legend()
        self.ax3.loglog()

        self.canvas.draw()

    def raise_compare(self):
        # Starts compare page
        images_dict = self.images
        frame = ComparisonPage(self.parent, self.controller, images_dict)
        frame.grid(row=0, column=0, sticky='nsew')
        frame.tkraise()
        self.controller.show_frame(AnalysisPage)

    def on_key_press(self, event):
        # Binder for matplotlib
        if event.key == 's':
            print("Saving plots...")
        key_press_handler(event, self.canvas, self.toolbar)
    
    def grab_tuple(self, s):
        # Helper for metadata parsing for ROI center
        s = s.split('(')[1]
        s = s.split(')')[0]
        x = int(s.split(',')[0])
        y = int(s.split(',')[1])
        return x,y


class ComparisonPage(tk.Frame):
    # Allows for comparison of two datasets, image set vs. image set or QPD data, with any combination of analysis techniques
    def __init__(self, parent, controller, images_dict):

        # Initialize frames for GUI elements
        tk.Frame.__init__(self, parent)
        self.newWindow = tk.Toplevel(self)
        self.newWindow.geometry('600x800')

        label = tk.Label(self.newWindow, text="Comparison", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.images = images_dict
        self.imageDir = os.getcwd()

        self.imageDir_lbl = tk.Label(self.newWindow, text='Current trial dir: ' + self.imageDir)
        self.imageDir_lbl.pack()

        # Initialize all widgets for frame
        frame1 = tk.Frame(self.newWindow)
        frame1.pack()

        back_btn = tk.Button(frame1, text="BACK", command=lambda: controller.show_frame(AnalysisPage))
        back_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        chg_dir_btn = tk.Button(frame1, text="Change directory", command=self.change_trialpath)
        chg_dir_btn.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        ttk.Separator(frame1, orient=tk.HORIZONTAL).grid(row=1, column=0, columnspan=99, sticky=(tk.W, tk.E))

        two_sets_btn = tk.Button(frame1, text='Two Datasets', command=self.two_sets_select)
        two_sets_btn.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)

        qpd_btn = tk.Button(frame1, text='QPD and Dataset', command=self.qpd)
        qpd_btn.grid(row=2, column=1, sticky='nsew', padx=5, pady=5)
  
    def change_trialpath(self):
        # Called on change directory button, handles selection and sets attribute for access across methods
        try:
            self.imageDir = filedialog.askdirectory(initialdir=os.getcwd())
            os.chdir(self.imageDir)
            self.imageDir_lbl.config(text='Current trial dir: ' + self.imageDir)
            self.imageDir_lbl.update_idletasks()
        except Exception as e:
            print('change_trialpath() ERROR: ', e)
    
    def two_sets_select(self):
        # Called on'Two Datasets' button, pops up new window for either porting from analysis or selecting two different trials

        if self.images is not None: # No analysis data loaded
            self.select_master = tk.Tk()
            self.select_master.geometry('100x100')
            self.select_master.title("Two Dataset Options")

            port = tk.Button(self.select_master, text='Port from Analysis', command=self.two_sets_port)
            port.pack()
            load = tk.Button(self.select_master, text='Load from directories', command=self.two_sets_load)
            load.pack()
        
            self.select_master.mainloop()
        else:
            self.two_sets_load()

    def two_sets_load(self):
        # Main for populating GUI widgets for selecting two separate trial datasets

        try:
            self.select_master.destroy() # Destroy option window
        except:
            pass

        optionframe = tk.Frame(self.newWindow)
        method_types = ['argmax', 'cv_com', 'gaussian', 'intensity', 'phase correlation']

        # Populate all GUI elements
        self.trial1_dir_btn = tk.Button(optionframe, text='Select first', bg='#fa8072', command=lambda: self.dir_select_helper(trial=1))            

        self.method1_var = tk.StringVar(optionframe)
        self.method1_var.set(method_types[0])
        method1_menu = tk.OptionMenu(optionframe, self.method1_var, *method_types)

        self.trial2_dir_btn = tk.Button(optionframe, text='Select second', bg='#fa8072', command=lambda: self.dir_select_helper(trial=2))

        self.method2_var = tk.StringVar(optionframe)
        self.method2_var.set(method_types[1])
        method2_menu = tk.OptionMenu(optionframe, self.method2_var, *method_types)

        self.trial1_dir_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)    
        method1_menu.grid(row=0, column=1, sticky='nsw', padx=5, pady=5)
        self.trial2_dir_btn.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        method2_menu.grid(row=0, column=3, sticky='nsw', padx=5, pady=5)

        optionframe.pack()

        # Start button to initialize graphing and comparisons
        self.start_frame = tk.Frame(self.newWindow)
        start_btn = tk.Button(self.start_frame, text='START', command=self.two_sets_load_data)
        start_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.start_frame.pack()
    
    def dir_select_helper(self, trial=1):
        # Called from two_sets_load, grabs directory for either dataset and sets button to green

        if trial == 1:
            self.trial1_dir = tk.filedialog.askdirectory(initialdir=self.imageDir)
            self.trial1_dir_btn.config(bg='#90ee90')
            self.trial1_dir_btn.update_idletasks()
        elif trial == 2:
            self.trial2_dir = tk.filedialog.askdirectory(initialdir=self.imageDir)
            self.trial2_dir_btn.config(bg='#90ee90')
            self.trial2_dir_btn.update_idletasks()
        elif trial == 3: # h5 file
            self.trial2_fname = tk.filedialog.askopenfilename(initialdir=self.imageDir)
            self.trial2_dir_btn.config(bg='#90ee90')
            self.trial2_dir_btn.update_idletasks()

    def two_sets_load_data(self):
        # Called from 'Start' button in two_sets_load, populates image list and calls comparison_graphs to handle graphing
        global loaded_dict

        # Load first dataset
        trial1_split = self.trial1_dir.split('trial')
        # Check if dataset has already been loaded, otherwise load and add
        if trial1_split[0][0:-1] in loaded_dict.keys():
            print('First set already loaded: {}'.format(self.trial1_dir))
            images1 = loaded_dict[trial1_split[0][0:-1]]['trial' + trial1_split[1]]
        else:
            os.chdir(self.trial1_dir)
            print('Loading first: {}'.format(self.trial1_dir))
            images1 = cca.load_images(cca.create_image_path())
            loaded_dict[trial1_split[0][0:-1]] = {}
            loaded_dict[trial1_split[0][0:-1]]['trial' + trial1_split[1]] = images1

        metadata1 = self.grab_metadata(self.trial1_dir)

        # Load second dataset
        trial2_split = self.trial2_dir.split('trial')
        # Check if dataset has already been loaded, otherwise load and add
        if trial2_split[0][0:-1] in loaded_dict.keys():
            print('Second set already loaded: {}'.format(self.trial2_dir))
            images2 = loaded_dict[trial2_split[0][0:-1]]['trial' + trial2_split[1]]
        else:
            os.chdir(self.trial2_dir)
            print('Loading second: {}'.format(self.trial2_dir))
            images2 = cca.load_images(cca.create_image_path())
            loaded_dict[trial2_split[0][0:-1]] = {}
            loaded_dict[trial2_split[0][0:-1]]['trial' + trial2_split[1]] = images2

        metadata2 = self.grab_metadata(self.trial2_dir)

        print('Datasets Loaded!')
        print()

        os.chdir(self.imageDir)

        # Unpack
        frame_rate1, duration1, roi_size1, roi_center1 = metadata1
        frame_rate2, duration2, roi_size2, roi_center2 = metadata2

        # Populate data
        os.chdir(self.trial1_dir)
        data1 = cca.data_analysis(images1, mean=self.method1_var.get(), frame_rate=frame_rate1, roi_center=roi_center1, roi_size=roi_size1)
        os.chdir(self.trial2_dir)
        data2 = cca.data_analysis(images2, mean=self.method2_var.get(), frame_rate=frame_rate2, roi_center=roi_center2, roi_size=roi_size2)

        # Set title
        if duration1 == duration2:
            title = "{} fps VS {} fps for {} sec with {}-wide ROI".format(frame_rate1, frame_rate2, duration1, roi_size1)
        else:
            title = "{} fps for {} sec VS {} fps for {} sec with {}-wide ROI". format(frame_rate1, duration1, frame_rate2, duration2, roi_size1)

        if self.method1_var.get() != 'intensity' and self.method2_var.get() != 'intensity':
            dtype = 'means'
        elif self.method1_var.get() == 'intensity' and self.method2_var.get() == 'intensity':
            dtype = 'intensity'
        else:
            dtype = 'mixed'

        self.comparison_graphs(data1, data2, title, dtype)

    def two_sets_port(self):
        # Main for populating GUI widgets for porting data from analysis page

        self.select_master.destroy() # Destroy option window

        # Initialize frame and lists
        optionframe = tk.Frame(self.newWindow)
        trialnums = [num for num in range(1, len(self.images)+1)]
        method_types = ['argmax', 'cv_com', 'gaussian', 'intensity', 'phase correlation']

        # First trial label, trialnum menu, analysis method menu
        self.trial1_intvar = tk.IntVar(optionframe)
        self.trial1_intvar.set(trialnums[0])
        trial1_menu = tk.OptionMenu(optionframe, self.trial1_intvar, *trialnums)

        trial1_lbl = tk.Label(optionframe, text='First dataset: trial_')

        self.method1_var = tk.StringVar(optionframe)
        self.method1_var.set(method_types[0])
        method1_menu = tk.OptionMenu(optionframe, self.method1_var, *method_types)

        # Second trial label, trialnum menu, analysis method menu
        self.trial2_intvar = tk.IntVar(optionframe)
        self.trial2_intvar.set(trialnums[1])
        trial2_menu = tk.OptionMenu(optionframe, self.trial2_intvar, *trialnums)

        trial2_lbl = tk.Label(optionframe, text='Second dataset: trial_')

        self.method2_var = tk.StringVar(optionframe)
        self.method2_var.set(method_types[1])
        method2_menu = tk.OptionMenu(optionframe, self.method2_var, *method_types)

        # Geometry manager for widgets
        trial1_lbl.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        trial1_menu.grid(row=0, column=1, sticky='nsw', padx=5, pady=5)
        method1_menu.grid(row=0, column=2, sticky='nsw', padx=5, pady=5)
        trial2_lbl.grid(row=0, column=3, sticky='nsew', padx=5, pady=5)
        trial2_menu.grid(row=0, column=4, sticky='nsw', padx=5, pady=5)
        method2_menu.grid(row=0, column=5, sticky='nsw', padx=5, pady=5)

        optionframe.pack()

        # Grab metadata and start button
        self.start_frame = tk.Frame(self.newWindow)

        metadata1 = self.grab_metadata(os.getcwd() + '/trial_{}'.format(self.trial1_intvar.get()))
        metadata2 = self.grab_metadata(os.getcwd() + '/trial_{}'.format(self.trial2_intvar.get()))

        pos_btn = tk.Button(self.start_frame, text='Start', command=lambda: self.two_sets_port_data(metadata1, metadata2))
        pos_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        self.start_frame.pack()
    
    def two_sets_port_data(self, metadata1, metadata2):
        # Called from 'Start' button in two_sets_port, populates data and calls comparison_graphs to handle graphing
        # Outdated function, will remove in future versions of code

        frame_rate1, duration1, roi_size1, roi_center1 = metadata1
        frame_rate2, duration2, roi_size2, roi_center2 = metadata2

        os.chdir(self.trial1_dir)
        print(self.trial1_dir, os.getcwd())
        data1 = cca.data_analysis(self.images['trial_{}'.format(self.trial1_intvar.get())], mean=self.method1_var.get(), frame_rate=frame_rate1, roi_center=roi_center1, roi_size=roi_size1)
        os.chdir(self.trial2_dir)
        print(self.trial2_dir, os.getcwd())
        data2 = cca.data_analysis(self.images['trial_{}'.format(self.trial2_intvar.get())], mean=self.method2_var.get(), frame_rate=frame_rate2, roi_center=roi_center2, roi_size=roi_size2)

        # fix this
        if self.method1_var.get() != 'intensity' and self.method2_var.get() != 'intensity':
            dtype = 'means'
        elif self.method1_var.get() == 'intensity' and self.method2_var.get() == 'intensity':
            dtype = 'intensity'
        else:
            dtype = 'mixed'
        title = 'TEST TITLE: ' + ': {} fps for {} sec with {} pixel-wide ROI'.format(frame_rate1, duration1, str(roi_size1))

        self.comparison_graphs(data1, data2, title, dtype)

    def grab_metadata(self, trialdir):
        # Helper function to parse metadata text file

        try:
            os.chdir(trialdir)
            files = os.listdir(os.getcwd())
            metadata_file = [f for f in files if f == 'metadata.txt'][0]
            metadata = open(metadata_file, 'r')
            for line in metadata:
                attr = line.split(':')
                if attr[0] == 'Frame Rate':
                    frame_rate = int(attr[1].strip())
                if attr[0] == 'Duration':
                    duration = attr[1].strip()
                if attr[0] == 'ROI size':
                    roi_size = int(attr[1].strip())
                if attr[0] == 'ROI center':
                    roi_center = self.grab_tuple(attr[1].strip())
            os.chdir('..')            
            return (frame_rate, duration, roi_size, roi_center)
        except:
            print('Error in metadata parsing')

    def grab_tuple(self, s):
        # Helper function to grab x,y ROI position in grab_metadata
        s = s.split('(')[1]
        s = s.split(')')[0]
        x = int(s.split(',')[0])
        y = int(s.split(',')[1])
        return x,y
        
    def qpd(self):
        # Main for comparing image set to QPD h5 file

        optionframe = tk.Frame(self.newWindow)
        method_types = ['argmax', 'cv_com', 'gaussian', 'intensity', 'phase correlation']

        # Populate all GUI elements
        self.trial1_dir_btn = tk.Button(optionframe, text='Select dataset', bg='#fa8072', command=lambda: self.dir_select_helper(trial=1))            

        self.method1_var = tk.StringVar(optionframe)
        self.method1_var.set(method_types[0])
        method1_menu = tk.OptionMenu(optionframe, self.method1_var, *method_types)

        self.trial2_dir_btn = tk.Button(optionframe, text='Select QPD .h5', bg='#fa8072', command=lambda: self.dir_select_helper(trial=3))

        self.trial1_dir_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)    
        method1_menu.grid(row=0, column=1, sticky='nsw', padx=5, pady=5)
        self.trial2_dir_btn.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)

        optionframe.pack()

        self.start_frame = tk.Frame(self.newWindow)
        start_btn = tk.Button(self.start_frame, text='START', command=self.qpd_load_data)
        start_btn.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.start_frame.pack()

    def qpd_load_data(self):
        
        global loaded_dict
        
        # Load first dataset
        trial1_split = self.trial1_dir.split('trial')

        # Check if dataset has already been loaded, otherwise load and add
        if trial1_split[0][0:-1] in loaded_dict.keys():
            print('First set already loaded: {}'.format(self.trial1_dir))
            images1 = loaded_dict[trial1_split[0][0:-1]]['trial' + trial1_split[1]]
        else:
            os.chdir(self.trial1_dir)
            print('Loading first: {}'.format(self.trial1_dir))
            images1 = cca.load_images(cca.create_image_path())
            loaded_dict[trial1_split[0][0:-1]] = {}
            loaded_dict[trial1_split[0][0:-1]]['trial' + trial1_split[1]] = images1

        metadata1 = self.grab_metadata(self.trial1_dir)

        # Unpack
        frame_rate1, duration1, roi_size1, roi_center1 = metadata1

        # Populate data
        data1 = cca.data_analysis(images1, mean=self.method1_var.get(), frame_rate=frame_rate1, roi_center=roi_center1, roi_size=roi_size1)
        
        fname = self.trial2_fname # filename
        bd = BeadDataFile.BeadDataFile(fname) #h5.wrapper written by Nadav
        x = bd.x2 # x coordinate, invisble for us
        y = bd.y2 # y coordinate
        z = bd.z2 # z coordinate
        qpd_rate = bd.fsamp
        print(os.getcwd())
        np.save('QPD.npy', ((x,y,z),qpd_rate))

        # Compute PSD for y,z QPD data
        freqs = rfftfreq(len(y), d=1./qpd_rate)
        norm = np.sqrt(2/(len(y)*qpd_rate))
        y_fft = rfft(y)
        y_psd = np.sqrt(norm**2 * (y_fft * y_fft.conj()).real)
        z_fft = rfft(z)
        z_psd = np.sqrt(norm**2 * (z_fft * z_fft.conj()).real)

        qpd_data = ((y,z), (freqs, y_psd, z_psd))

        # Set dtype flag for comparison_graphs function
        if self.method1_var.get() == 'intensity':
            dtype = 'mixed'
        else:
            dtype = 'means'

        self.comparison_graphs(data1, qpd_data, 'QPD Comparisons', dtype)

    def swap(self, data1, data2, title, dtype):
        # Helper to swap inputs from comparison_graphs
        self.comparison_graphs(data2, data1, title, dtype)

    def comparison_graphs(self, data1, data2, title, dtype):
        # Handles graphing of data passed from various data population functions
        # params data1/2: either (means_lst, psd_tuple) or (vals_lst, freqs_lst, psd_lst) for intensity
        # param title: title for graph
        # param dtype: flag (string) for what comparison method to use depending on input data

        self.graph_frame = tk.Frame(self.newWindow)
        self.graph_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        # Handle population/deletion of swap input button (gets tricky in recursive call)
        try: 
            self.swap_btn.grid_forget()
        except:
            swap_btn = tk.Button(self.start_frame, text='Swap Inputs', command=lambda: (self.swap(data1, data2, title, dtype)))
            swap_btn.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)

        # Initialize graph canvas and figure
        try:
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
        except: # On first call
            self.fig = Figure(figsize=(5,6), dpi=100)
            self.fig.subplotpars.hspace = 0.5

            self.ax1 = self.fig.add_subplot(311)
            self.ax2 = self.fig.add_subplot(312)
            self.ax3 = self.fig.add_subplot(313)

            self.vbar = tk.Scrollbar(self.graph_frame, orient=tk.VERTICAL)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            self.canvas.get_tk_widget().config(width=500, height=1000, scrollregion=(0,0,800,800))
            self.canvas.get_tk_widget().config(yscrollcommand=self.vbar.set)
            self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

            self.vbar.pack(side=tk.RIGHT, fill=tk.Y, expand=1)
            self.vbar.config(command=self.canvas.get_tk_widget().yview)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()
            self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        if dtype=='means': # Means data
            means1, psd1 = data1
            means2, psd2 = data2
            freqs1, xpsd1, ypsd1 = psd1
            freqs2, xpsd2, ypsd2 = psd2

            # Normalize lists to smallest frequency spectrum
            if len(freqs1) != len(freqs2):
                print('Different input lengths... Normalizing inputs')
                if len(freqs1) < len(freqs2):
                    freqs = freqs1
                    xpsd2 = xpsd2[0:len(xpsd1)]
                    ypsd2 = ypsd2[0:len(ypsd1)]
                else:
                    freqs = freqs2
                    xpsd1 = xpsd1[0:len(xpsd2)]
                    ypsd1 = ypsd1[0:len(ypsd2)]
            else:
                freqs = freqs1

            # Grab comparison data
            xpsd_diff, xpsd_ratio = cca.compare_signals(xpsd1, xpsd2)
            ypsd_diff, ypsd_ratio = cca.compare_signals(ypsd1, ypsd2)            

            # Plotting
            self.fig.suptitle(title)
            self.ax1.loglog(freqs, xpsd1, label='y1')
            self.ax1.loglog(freqs, xpsd2, label='y2')
            self.ax1.loglog(freqs, ypsd1, label='z1')
            self.ax1.loglog(freqs, ypsd2, label='z2')
            self.ax1.set_title("Position PSDs")
            self.ax1.set(xlabel="Frequency [Hz]", ylabel='PSD [arb.]')

            self.ax2.loglog(freqs, xpsd_diff, label='difference')
            self.ax2.loglog(freqs, xpsd_ratio, label='ratio')
            self.ax2.set_title("y_PSDs")
            self.ax2.set(xlabel="Frequency [Hz]", ylabel="Diff/Ratio")

            self.ax3.plot(freqs, ypsd_diff, label='difference')
            self.ax3.plot(freqs, ypsd_ratio, label='ratio')
            self.ax3.set_title("z_PSDs")
            self.ax3.set(xlabel="Frequency [Hz]", ylabel="Diff/Ratio")

            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()

            self.canvas.draw()

        elif dtype=='intensity': # Intensity comparison

            vals1, freqs1, psd1 = data1
            vals2, freqs2, psd2 = data2

            # Normalize lists to smallest frequency spectrum
            if len(freqs1) != len(freqs2):
                print('Different input lengths... Normalizing inputs')
                if len(freqs1) < len(freqs2):
                    freqs = freqs1
                    vals2 = vals2[0:len(vals1)]
                    psd2 = psd2[0:len(psd1)]
                else:
                    freqs = freqs2
                    vals1 = vals1[0:len(vals2)]
                    psd1 = psd1[0:len(psd2)]
            else:
                freqs = freqs1

            diff, ratio = cca.compare_signals(psd1, psd2)

            self.fig.suptitle(title)
            self.ax1.plot(np.arange(len(vals1)), vals1, label='1')
            self.ax1.plot(np.arange(len(vals2)), vals2, label='2')
            self.ax1.set_title("Intensities")
            self.ax1.set(xlabel="Frame num", ylabel="Intensity Value")

            self.ax2.set_title("Intensity PSDs")
            self.ax2.set(xlabel="Frequency [Hz]", ylabel="PSD [arb.]")
            self.ax2.loglog(freqs, psd1, label='psd1')
            self.ax2.loglog(freqs, psd2, label='psd2')

            self.ax3.set_title("Difference/Ratio")
            self.ax3.set(xlabel="Frequency [Hz]", ylabel="Diff")
            self.ax3.set(xlabel="Frequency [Hz]", ylabel="Ratio")
            self.ax3.loglog(freqs, diff, label='Difference')
            self.ax3.loglog(freqs, ratio, label='Ratio')
            
            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()

            self.canvas.draw()

        elif dtype=='mixed': # Intensity vs means
            # Unpack data
            try:
                vals, ifreqs, ipsd = data1
                means, means_psd = data2
            except:
                vals, ifreqs, ipsd = data2
                means, means_psd = data1
            means = list(means)
            
            mfreqs, ypsd, zpsd = means_psd

            # Normalize
            if len(mfreqs) != len(ifreqs):
                print('Different input lengths... Normalizing inputs')
                if len(mfreqs) < len(ifreqs):
                    freqs = mfreqs
                    vals = vals[0:len(means[0])]
                else:
                    freqs = ifreqs
                    means[0] = means[0][0:len(vals)]
                    means[1] = means[1][0:len(vals)]
            else:
                freqs = mfreqs

            # Compare
            ydiff, yratio = cca.compare_signals(ypsd, ipsd)
            zdiff, zratio = cca.compare_signals(zpsd, ipsd)

            # Plot
            self.fig.suptitle(title)
            self.ax1.plot(np.arange(len(means[0])), means[0], label='y')
            self.ax1.plot(np.arange(len(means[1])), means[1], label='z')
            self.ax1.plot(np.arange(len(vals)), vals, label='intensity')
            self.ax1.set_title("Intensity/Means")
            self.ax1.set(xlabel="Frame num", ylabel="Pixel num / Intensity")

            self.ax2.loglog(freqs, ydiff, label='Difference')
            self.ax2.loglog(freqs, yratio, label='Ratio')
            self.ax2.set_title("y Difference/Ratio")
            self.ax2.set(xlabel="Frequency [Hz]", ylabel="Diff/Ratio")

            self.ax3.loglog(freqs, zdiff, label='Difference')
            self.ax3.loglog(freqs, zratio, label='Ratio')
            self.ax3.set_title("z Difference/Ratio")
            self.ax2.set(xlabel="Frequency [Hz]", ylabel="Diff/Ratio")
            self.ax3.set(xlabel="Frequency [Hz]", ylabel="Diff/Ratio")

            self.ax1.legend()
            self.ax2.legend()
            self.ax3.legend()

            self.canvas.draw()

    def on_key_press(self, event):
        if event.key == 's':
            print("Saving plots...")
        key_press_handler(event, self.canvas, self.toolbar)
        

if __name__ == '__main__':  
    window = analysis_GUI()
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    window.mainloop()