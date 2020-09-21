IMAGE ANALYSIS GUI
Emmett Hough, Gratta Gravity Group
Summer 2020

1. Introduction

The motivation behind this project is to provide an independent system for measuring and analysing levitated microspheres. The current position feedback systems are complicated and coupled to the trapping laser and optics, and so by adding an independent camera looking directly into the trap, we hope to not only provide easy position and frequency analysis, but also provide another independent, inclusive system for tracking microspheres.

The idea for this GUI is to have an all-inclusive program to control data capture from the camera, store that data, provide quick analysis using various position and measurement algorithms, as well as give a brief comparison between different datasets. The GUI saves all analysis scripts as numpy files with the intention that any serious analysis will be done in a jupyter notebook or likewise, as the GUI window itself isn't very controllable as far as bounds and resizing, but is perfect for seeing leading-order behavior.

2. Page Breakdowns and Useage

The GUI has three main pages and a start page, each with a distinct function. The Start Page is self-explainatory, it provides navigation between the three major pages and nothing else.

ACQUISITION PAGE

This is where the GUI talks to the actual camera and captures data. You can change the data directory by clicking the "Change working directory" button. In the Controls panel, you can input frame rate, duration (in seconds), and the trial name, as well as selecting a Region of Interest (ROI) and its size, which is set as a box around the brightest pixel of a test frame. Clicking the bead height button will take a test frame and print out the x,y pixel of the detected bead using argmax, and if the checkbox is selected, the frame will be displayed with a dot at the detected point.

The console output provides useful information about where the data is being saved, when ROI is set and frame capture begins, and a confirmation of the number of captured frames. It should be noted that the GUI is meant to be run in conjunction with the terminal window running the script, since writing all output to a tkinter console would be impractical, so more information (especially in the Analysis and Comparison pages) will print out to the console as the script runs.

Directory handling protocol:
Working directory -> trial name directory -> trial number directory -> numpy files of frame data and metadata txt files

The trial number subdirectories are so that one can capture multiple trials with the same frame rate and duration without changing any settings.

If the 'Show analysis after' checkbox is selected, the GUI pulls up the Analysis page with the trial directory data pre-loaded.

ANALYSIS PAGE

This page is for getting a first glance at captured data. The first thing that must be done (unless porting from acquisition) is to click the 'LOAD DATA' button, where you must select a trial directory, NOT a trial number sub-directory. If appropriate trial naming is used, this shouldn't be hard, just select the folder containing the trial_0, trial_1, ... subdirectories. The console will output a loading bar as the program reads in the frame data (be patient, thousands of frames can take a bit to load up). However once a dataset is loaded, it's stored in the program for fast access within a single session of the GUI. This may seem trivial, but it saves a lot of time when comparing a dataset in the comparison page.

Once the dataset is loaded, trial num buttons appear, along with the 'COMPARE' button. Select which position algorithm you want to use from the dropdown menu, and click on a trial num button to pull up the raw bead position data, the corresponding Power Spectral Densities (PSDs), and Intensity PSD. A 'Single-Pixel Intensity' button also appears after graphs are populated (explained below). Standard matplotlib key bindings are implemented, so to save the figure simply press 's' on the keyboard.

Analysis methods: 
1. argmax -> selects brightest pixel of the frame, which is fast but inefficient and limited in resolution
2. cv_com -> uses cv2's centroid algorithm to get a 'center of mass' of pixel values
3. gaussian -> fits a gaussian to the row and column means, of which the sigma is used as bead position
4. phase correlation -> uses 2D cross-correlated gaussians to compute shift between two images, which is incredibly powerful and accurate. no reason not to use this all the time, other than a bit of time for computation. Documentation: https://scikit-image.org/docs/0.13.x/auto_examples/transform/plot_register_translation.html
5. intensity -> takes mean of each frame as a data point, then takes a PSD to see periodic behavior in overall intensity of the images
6. single-pixel intensity -> samples N pixels from either a guassian or random distribution across both axes of the frame, for each pixel we take that pixel's value through the dataset (i.e. pixel value at (15,23) for all 8000 frames of a set) as input to a PSD, then average across the N pixels. The idea with gaussian samples is that you weight the PSD of the bead (which is ideally at the center of the ROI) but also get contributions from background fluctuations.

COMPARISON PAGE

This page is meant to provide a quick way to look at two datasets side-by-side, however in practice the graphs are too small and crowded to see anything meaningful. It's a good way to see leading-order behavior, as well as to compare to QPD data, but any real analysis should be done offline.

That being said, when the comparison page is opened you can either load two image datasets or compare a dataset to QPD data. If the comparison page was accessed from the analysis page, a window will pop up upon pressing the two datasets button, asking to either port from analysis or load from directories. If the comparison page is accessed directly from the start page, this option is not available. Porting from analysis will automatically load in the dataset that was being analyzed and give dropdown menus to select which two subtrials you want to compare, and with which analysis method. The idea here is to be able to compare slight differences within one trial between trial nums, or compare one trial num with two different analysis techniques.

Loading from diffent datasets populates the GUI with slightly different elements, two select buttons and two analysis method dropdowns. Click the first and second select buttons to tell the GUI which subtrial nums (NOTE: this is different from loading data in the analysis page. Here you want to select the trial_* subdirectory. Sorry.) that you want to compare.

Comparing a QPD .h5 dataset to an image dataset is very similar to loading two datasets. Make sure the QPD dataset is a .h5 file, and select the actual file from the dialouge that pops up.

Pressing 'START' populates the comparison graphs, which differ based on what you're comparing (position vs position, position vs intensity, or intensity vs intensity). Again, saving plots is done by pressing 's'. The plots here are simply the raw PSDs, then the y and z axis ratio and difference between the two input datasets. The inputs can be swapped by pressing the 'Swap Inputs' button, which only really affects the ratio plots.


3. Conclusions

Overall, the GUI works really well. Acquisition is simple and accessable, and at-a-glance analysis is right at your fingertips. As a preliminary test of the GUI and image analysis techniques, the bead was imaged while in the high-pressure regime of the trap while exhibiting damped, randomly-driven harmonic motion (overdamped in z, underdamped in y), and using the phase cross correlation method, the expected spectra were extracted directly from image data. This proves that the camera GUI system is useful for analysis, and the next steps in development include automating capture, calibrating with the QPD, and imaging smaller amplitudes of motion to test lower limits of the system. Overall I'm very happy with how this turned out, and I'm happy to answer any questions at emhough@stanford.edu
