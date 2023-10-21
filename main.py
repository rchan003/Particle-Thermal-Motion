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

num_files = 29
num_rows = 119
sum_list = [0] * 119

# creating string list of all files containing distance data
os.chdir(str(Path.cwd())+'/thermal_motion/distance_data')
dist_files = glob.glob("*.txt")
file = dist_files[0]
data = pd.read_csv(file, header=None)
data.columns = ['d']

# creating a list of ALL the distances
all_distances = []

i = 0
j = 0
for i in range(num_files):
    read_file = dist_files[i]
    data = pd.read_csv(read_file, header=None)
    data.columns = ['d']
    for j in range(num_rows):
        sum_list[j] = data['d'][j] + sum_list[j]
        all_distances.append(data['d'][j])

print('Number Datapoints: ',len(all_distances))

# removing outliers
data_removed_counter = 0
for i in range(0, len(all_distances)):
    if all_distances[i] == 0:
        all_distances.remove(all_distances[i])
        data_removed_counter += 1

    if all_distances[i] >= 30:
        all_distances.remove(all_distances[i])
        data_removed_counter += 1

print(data_removed_counter, "points removed out of", len(all_distances)+data_removed_counter)

# converting to average displacement in micrometers and total distance traveled in micrometers
total_distance_list = [None] * num_rows
delta_t = 0.5
time_axis = [None] * num_rows
average_list = [None] * num_rows
i = 0
for i in range(num_rows):
    average_list[i] = sum_list[i] * (1/num_files)
    time_axis[i] = delta_t * i
    if i == 0:
        total_distance_list[i] = average_list[i]
    else:
        total_distance_list[i] = average_list[i] + total_distance_list[i-1]


# saving averages + distances
f = open(str(Path.cwd())+'/average.txt', 'w')
for ele in average_list:
    f.write("%f\n" % ele)
f.close()

f = open(str(Path.cwd())+'/all_distances.txt', 'w')
for ele in all_distances:
    f.write("%f\n" % ele)
f.close()

# Linear fit
lin_fit = scipy.stats.linregress(time_axis, total_distance_list)
m, b, r_val, chi_const = lin_fit[0], lin_fit[1], lin_fit[2], lin_fit[4]
lin_fit_vals = [m*time_axis[i]+b for i in range(0,len(time_axis))]

# Plotting
plt.text(15, 1, 'line of best fit: y=6.187t+8.307e-12')
plt.errorbar(time_axis, total_distance_list,
             xerr=0.03, yerr=0.1, label="data", zorder=1)
plt.plot(time_axis, lin_fit_vals, label="linear fit", zorder=2)
plt.legend(loc="upper left")
plt.title('Averaged Mean Squared Distance Travelled vs Time')
plt.ylabel('Mean Squared Distance [μm]')
plt.xlabel('Time [s]')
plt.show()


def func18(r, D):
    t = 0.5
    return (r/(2*D*t)) * np.exp(-(r*r)/(4*D*t))


# Constants for k
radius = 0.95 * math.pow(10, -6)
vis = 0.001
T = 296.5
k_accepted = 1.38 * math.pow(10, -23)
gamma = radius * vis * 4.0 * math.pi


# plotting histogram
bin_size = 250
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(sorted(all_distances), bins=bin_size)
plt.title("Probability Distribution of Step Lengths")
plt.xlabel("Displacement over 0.5 of a second [μm]")
plt.show()

# finding D and its error by curvefitting
hist_vals, bin_edges, ignore = plt.hist(
    sorted(all_distances), bins=250, density=True, stacked=True)
popt, pcov = scipy.optimize.curve_fit(func18, bin_edges[0:-1], hist_vals)
D_curvefit = popt[0]
D_err = np.sqrt(pcov[0][0])

k_curvefit = D_curvefit * math.pow(10, -12)*gamma/T

print("\nD_curvefit: ", D_curvefit, "\nD_curvefit_err: ",
      D_err, "\nk_curvefit: ", k_curvefit)

# using estimate to find k
estimate = np.sum(np.square(all_distances)) / (2*len(all_distances))
print("Maximum likelihood estimate: ", estimate)

D_estimate = estimate / 2
k_estimate = estimate * gamma * math.pow(10, -12) / (2*T)
print("k_estimate: ", k_estimate, "\nD_estimate: ", D_estimate)

percent_diff_k_estimate = abs(k_accepted-k_estimate)/k_accepted * 100
print("Percent difference between k_estimate and k_actual: ",
      percent_diff_k_estimate)

# k from linfit
k_linfit = m*math.pow(10, -12)*gamma/(4*T)
D_linfit = k_linfit*T/gamma * math.pow(10, 12)
print("k_linfit:", k_linfit, "\nD_linfit: ", D_linfit)

percent_diff_k_linfit = abs(k_accepted-k_linfit)/k_accepted * 100
print("Percent difference between k_linfit and k_actual: ",
      percent_diff_k_linfit)

# final histogram using all the different estimates of k
sorted_distances = sorted(all_distances)
x_axis = np.linspace(min(sorted_distances), max(sorted_distances), 10000)
y_axis_curvefit = [func18(i, D_curvefit) for i in x_axis]
y_axis_linfit = [func18(i, D_linfit) for i in x_axis]
y_axis_estimate = [func18(i, D_estimate) for i in x_axis]
plt.plot(x_axis, y_axis_curvefit, '-r', label="Curve fit estimate")
plt.plot(x_axis, y_axis_linfit, '-y', label="Linear fit estimate")
plt.plot(x_axis, y_axis_estimate, '-b', label="Maximum likelihood estimate")
plt.xlim([-1, 16])
plt.legend(loc="upper right")
plt.title('Histogram')
plt.xlabel('Distance Travelled [μm]')
plt.ylabel('Probability Density')
plt.show()


# calculating uncertainty for k_curvefit
position_err = 0.1*math.pow(10, -6)
time_err = 0.03
vis_err = 0.05*0.001
temp_err = 0.5

a = radius*vis*D_err/T
b = D_curvefit*vis*position_err/T
c = D_curvefit*radius*vis_err/T
d = D_curvefit*radius*vis*temp_err/(T**2)
k_curvefit_err = (
    4*math.pi*(((a)**2 + (b) ** 2 + (c)**2 + (d)**2)**0.5))*math.pow(10, -12)
percent_diff_k_curvefit = abs(k_accepted-k_curvefit)/k_accepted * 100

print("Percent difference between k_curvefit and k_actual: ",
      percent_diff_k_curvefit)

print("Uncertainty for k_curvefit: ", k_curvefit_err)

# chi squared (linear fit)
chi_lin_fit = [None] * len(total_distance_list)
residuals_lin_fit = [None] * len(total_distance_list)
lin_equation_array = [None] * len(total_distance_list)

for i in range(0, len(total_distance_list)):
    lin_equation_array[i] = m*(time_axis[i]) + b
    residuals_lin_fit[i] = lin_equation_array[i] - total_distance_list[i]
    chi_lin_fit[i] = ((residuals_lin_fit[i]**2) /
                      (2*chi_const*(len(total_distance_list)**0.5)**2))

print("m = ", m)
print("b = ", b)

# residual plot of linear fit
r_chi_lin = (sum(chi_lin_fit)/(len(chi_lin_fit)-1))**0.5
print("Reduced Chi square for the linear fit: ", r_chi_lin)
plt.scatter(time_axis, residuals_lin_fit)
plt.plot(time_axis, np.zeros(len(average_list)))
plt.xlabel("Time [s]")
plt.ylabel("Residuals")
plt.title("Residuals of Linear Fit")
plt.show()


hist_vals, bin_edges, ignore = plt.hist(
    sorted(all_distances), bins=250, density=True, stacked=True)

popt, pcov = scipy.optimize.curve_fit(func18, bin_edges[0:-1], hist_vals)
D_curvefit = popt[0]
D_err = np.sqrt(pcov[0][0])


j = 1
dist_axis_hist = [None] * 250
for i in range(0,250):
    dist_axis_hist[i] = (bin_edges[j] - bin_edges[i])/2
    j += 1

# max length
difference = max(sorted_distances) - min(sorted_distances)

# determining reduced chi squared and residuals
y_axis = []
for i in range(0,250):
    y_axis.append(func18(bin_edges[i], D_curvefit))
chi_curve = []
residuals_curve = []
for i in range(len(hist_vals)):
    try:
        residuals_curve.append(y_axis[i]-hist_vals[i])
        chi_curve.append((residuals_curve[i]**2)/D_err)
    except:
        continue

chi_curve_arr = np.array(chi_curve)
r_chi_curve = (sum(chi_curve)/(len(chi_curve)-1))
print("Reduced chi squared for curvefit: ", r_chi_curve)


# residuals plot
resid_len = len(residuals_curve)
fig = plt.figure()
plt.scatter(bin_edges[0:250], residuals_curve)
plt.plot(bin_edges[0:250], np.zeros(resid_len))
plt.title("Residuals of Curvefit")
plt.xlabel('Distance travelled [μm]')
plt.ylabel('Residuals')
plt.show()
