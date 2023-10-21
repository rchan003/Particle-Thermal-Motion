import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy import stats
import csv
import math
import glob
import os
from pathlib import Path
''' Computes the mean squared distance for each particle and stores it into text files in the all_RMS folder
also plots mean squared graph and its residuals
'''
num_files = 29
num_rows = 119
sum_list = [0] * 119

# creating string list of all files containing distance data
all_distances = pd.read_csv(str(Path.cwd()) + '/thermal_motion/distance_data/all_distances.txt', header=None)
os.chdir(str(Path.cwd()) + '/thermal_motion/all_RMS')

rms_files = glob.glob("*.txt")
all_distances.columns = ['d']


counter = 0
for i in range(0,num_files):
    file = open(rms_files[i], 'w')
    for j in range(0,num_rows):
        if j == 0:
            number = float(all_distances['d'][i*num_rows+j])
        else:
            number = float(all_distances['d'][i*num_rows+j]) + prev_number
        file.write(F'{number}\n')
        prev_number = number
    counter += 1


delta_t = 0.5
time_axis = [None] * num_rows
for i in range(0,num_rows):
    time_axis[i] = delta_t * i

for i in range(0,num_files):
    file = pd.read_csv(rms_files[i], header=None)
    file.columns = ['d']
    plt.plot(time_axis, file['d'])

m = 6.186642399460729
b = 8.307602801672788*math.pow(10, -12)
lin_fit = [None]*len(time_axis)
for i in range(0,len(time_axis)):
    lin_fit[i] = m*time_axis[i] + b

plt.plot(time_axis, lin_fit, color='black', label="Linear Fit for Average")
plt.legend(loc="upper left")
plt.text(15, 1, 'line of best fit: y=6.187t+8.307e-12')
plt.title("Mean Squared Distance vs Time")
plt.xlabel("Time [s]")
plt.ylabel("RMS Distance [μm]")
plt.show()



for i in range(0, num_files):
    file = pd.read_csv(rms_files[i], header=None)
    file.columns = ['d']
    residuals_lin_fit = [None] * len(file['d'])
    for j in range(0,len(file['d'])):
        #print(lin_fit[j] - file['d'][j])
        residuals_lin_fit[j] = lin_fit[j] - file['d'][j]
    plt.plot(time_axis, residuals_lin_fit, marker=".")

plt.plot(time_axis, [0]*len(time_axis), color="black")
plt.title("Residuals of Linear Fit")
plt.xlabel("Time [s]")
plt.ylabel("Residuals")
plt.show()
