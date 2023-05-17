import matplotlib.pyplot as plt
import numpy as np
import math
import os
from angle_toolbox import theta_calc, phi_calc
import sys
import time
import matplotlib.animation as animation
import matplotlib
from IPython.display import HTML

sys.setrecursionlimit(10000)

os.system('clear')

# Initialize current position tracker
current_pos = [0, 0, 0]

# Initialize 3D control point arrays
x = [current_pos[0]]
y = [current_pos[1]]
z = [current_pos[2]]
xBezier = []
yBezier = []
zBezier = []

# Trying classical following to start, drawing straight lines between points
x_classic = x.copy()
y_classic = y.copy()
z_classic = z.copy()

x_animation_array = []
y_animation_array = []
z_animation_array = []
xBezier_array = []
yBezier_array = []
zBezier_array = []
current_pos_array = []

# Initialize phi and theta array output
phi_array = []
theta_array = []

# Initialize actuation monitoring
actuations = 0
max_actuations = 32
link_length = 6

# Initialize maximum curvature
curvature_limit = math.pi/6

# Input points, handle errors as needed
while 1:
    try:
        os.system('clear')
        x.append(int(input("Input x coordinate of point: ")))
        y.append(int(input("Input y coordinate of point: ")))
        z.append(int(input("Input z coordinate of point: ")))
        os.system('clear')
    except:
        os.system('clear')
        print("There was an error in entering your point. You may have entered a letter. Program shutting down... ")
        exit()

    try:
        if input("Would you like to enter more points? y for yes: ").lower() != "y":
            os.system('clear')
            print("Point entry complete.")
            break
    except:
        os.system('clear')
        print(
            "There was an error in understanding your response. Continuing onwards.")
        break

def classic_follow():
    for i in range(1, len(x)):
        x_dif = x[i]-current_pos[0]
        y_dif = y[i]-current_pos[1]
        z_dif = z[i]-current_pos[2]
        vector_difference = [x_dif, y_dif, z_dif]
        mag_vector_difference = math.sqrt(x_dif**2+y_dif**2+z_dif**2)
        theta = theta_calc(
        (vector_difference[0], vector_difference[1], vector_difference[2]), (0, 1, 0))
        phi = phi_calc(
        (vector_difference[0], vector_difference[1], vector_difference[2]), (-1, 0, 0))
        n_actuations = math.ceil(mag_vector_difference/link_length)
        if phi<curvature_limit:
            current_pos[0]+=vector_difference[0]/mag_vector_difference*n_actuations
            current_pos[1]+=vector_difference[1]/mag_vector_difference*n_actuations
            current_pos[2]+=vector_difference[2]/mag_vector_difference*n_actuations
        else:
            break



classic_follow()