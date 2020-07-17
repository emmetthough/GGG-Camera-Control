# Analysis GUI script
# Emmett Hough, Summer Research 2020

import sys
import os
import traceback
from time import sleep
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings:
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import camera_control_analysis as cca

LARGE_FONT = ("Verdana 24 bold")
MEDIUM_FONT = ("Verdana 18")
ROI_size = 16

global path 
path = os.getcwd()

global imageDir

class analysis_GUI(tk.Tk):

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

        button = tk.Button(self, text="Acquisition", command=lambda: controller.show_frame(AcquisitionPage))
        button.pack()

        button2 = tk.Button(self, text="Analysis", command=lambda: controller.show_frame(AnalysisPage))
        button2.pack()


class AcquisitionPage(tk.Frame):

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
        show_chkbtn.grid(row=3, column=1, sticky='w', padx=5, pady=5)
        height_btn.grid(row=4, column=0, sticky='w', padx=5, pady=5)
        imshow_chkbtn.grid(row=4, column=1, sticky='w', padx=5, pady=5)
        start_btn.grid(row=5, column=0, sticky='nsew', padx=5, pady=5)

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
            
            cca.set_camera_defaults()

            # Set ROI if checkbutton is true
            if self.roi_var.get():
                self.listbox.insert(tk.END, 'Setting ROI...')
                self.listbox.update_idletasks()
                x, y = cca.set_roi(size=ROI_size)
                self.listbox.insert(tk.END, 'ROI ({} pixels wide) set: center at {}, {}'.format(ROI_size,x,y))
                self.listbox.update_idletasks()

            sleep(0.5)

            self.listbox.insert(tk.END, 'Starting frame capture')
            self.listbox.update_idletasks()
            cca.aquire_frames(framerate, duration, os.getcwd())
            cca.set_camera_defaults()

            num_frames = len(os.listdir(os.getcwd()))
            self.listbox.insert(tk.END, '{} frames successfully captured'.format(num_frames))
            self.listbox.insert(tk.END, '')
            self.listbox.update_idletasks()

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
        self.listbox.insert(tk.END, path)
        print('Acquisition path:', path)
        if path != os.getcwd():
            print("ERROR: directory failure. See change_path()")

    def bead_height(self):
        # Get position of bead with argmax method, either print out or show image
        cca.set_camera_defaults()
        x,y = cca.bead_height(imshow=self.imshow_var.get())
        if not self.imshow_var.get():
            self.listbox.insert(tk.END, "Bead position: {}, {}".format(x,y))
            self.listbox.update_idletasks()


class AnalysisPage(tk.Frame):

    def __init__(self, parent, controller):

        # Initialize with buttons
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Analysis", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        frame1 = tk.Frame(self)
        frame1.pack()

        button1 = tk.Button(frame1, text="Back to Home", command=lambda: controller.show_frame(StartPage))
        button1.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)

        button2 = tk.Button(frame1, text="Load from directory", command=self.change_trialpath)
        button2.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
    
        self.imageDir = os.getcwd()

        self.imageDir_lbl = tk.Label(self, text='Current trial dir: ' + self.imageDir)
        self.imageDir_lbl.pack()


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
        
        # Populate and load dictionary with numpy arrays from each trial in directory
        images = {}
        for trialnum in trials:
            os.chdir(trialnum)
            npfile_lst = cca.load_images(cca.create_image_path())
            os.chdir('..')
            images[trialnum] = npfile_lst

        # Make and populate buttons for each trial in directory
        # Initialize data for each button, which pulls up graphs upon call
        trialframe = tk.Frame(self)
        col = 0
        for trial in trials:
            but = tk.Button(trialframe, text=trial, command=lambda trial=trial: self.analysis_graphs(images[trial], trial))
            but.grid(row=0, column=col, sticky='nsew', padx=5, pady=5)
            col += 1
        trialframe.pack()
    
    def analysis_graphs(self, npfile_lst, trialnum):
        # Called on trial button push, handles canvas and graphing

        self.frame1 = tk.Frame(self)
        self.frame1.pack()

        try:
            self.ax1.clear()
            self.ax2.clear()
        except: # On first call
            self.fig = Figure(figsize=(6,10), dpi=100)

            # self.ax1.set_ymargin(m=0.5)
            # self.fig.subplotpars.top = 5
            # self.fig.subplotpars.bottom = 0.1
            self.fig.subplotpars.hspace = 0.5

            self.ax1 = self.fig.add_subplot(211)
            self.ax2 = self.fig.add_subplot(212)

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

        # Get data
        means, ffts = cca.data_analysis(npfile_lst)
        freqs, xfft, yfft = ffts

        # Plot!
        self.fig.suptitle(trialnum, fontweight='bold')
        self.ax1.plot(np.arange(len(means[0])), means[0], label='x')
        self.ax1.plot(np.arange(len(means[0])), means[1], label='y')
        self.ax2.plot(freqs, xfft, label='x')
        self.ax2.plot(freqs, yfft, label='y')
        self.ax1.set_title("Bead positions")
        self.ax2.set_title("Position FFTs")
        self.ax1.legend()
        self.ax2.loglog()
        self.ax2.legend()
        self.ax1.set(xlabel='test')
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
