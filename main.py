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

# Initialize 3D control point arrays
x = [0]
y = [0]
z = [0]
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

# Initialize current position tracker
current_pos = [0, 0, 0]

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


def classic_follow(x, y, z, current_pos, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array, theta_array, phi_array):
    xBezier_array.append(x)
    yBezier_array.append(y)
    zBezier_array.append(z)
    current_pos_array.append(current_pos)
    x_animation_array.append(x.copy())
    y_animation_array.append(y.copy())
    z_animation_array.append(z.copy())
    if len(x) == 1:
        print(f"Path following complete, current position is {current_pos}")
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.plot(
            x_classic, y_classic, z_classic, c='green')
        ax1.set_xlabel("x axis")
        ax1.set_ylabel("y axis")
        ax1.set_zlabel("z axis")

        def animate(i):
            ax1.cla()
            ax1.scatter(
                x_animation_array[i], y_animation_array[i], z_animation_array[i], c='black')
            current_pos_x = []
            current_pos_y = []
            current_pos_z = []
            for j in range(0, i):
                current_pos_x.append(current_pos_array[j][0])
                current_pos_y.append(current_pos_array[j][1])
                current_pos_z.append(current_pos_array[j][2])
            ax1.plot(
                current_pos_x, current_pos_y, current_pos_z, c='blue', linewidth=5)
            ax1.plot(
                xBezier_array[i], yBezier_array[i], zBezier_array[i], c='red')
            fig.canvas.draw_idle()

        n_frames = len(xBezier_array)
        ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                 frames=range(0, n_frames), interval=300, repeat=True)
        plt.show()
        return 0
    elif actuations > max_actuations:
        print(f"Snake fully extended, current position is {current_pos}")
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(111, projection='3d')

        ax1.plot(
            x_classic, y_classic, z_classic, c='green')
        ax1.set_xlabel("x axis")
        ax1.set_ylabel("y axis")
        ax1.set_zlabel("z axis")

        def animate(i):
            ax1.cla()
            ax1.scatter(
                x_animation_array[i], y_animation_array[i], z_animation_array[i], c='black')
            current_pos_x = []
            current_pos_y = []
            current_pos_z = []
            for j in range(0, i):
                current_pos_x.append(current_pos_array[j][0])
                current_pos_y.append(current_pos_array[j][1])
                current_pos_z.append(current_pos_array[j][2])
            ax1.plot(
                current_pos_x, current_pos_y, current_pos_z, c='blue', linewidth=5)
            ax1.plot(
                xBezier_array[i], yBezier_array[i], zBezier_array[i], c='red')
            fig.canvas.draw_idle()

        n_frames = len(xBezier_array)
        ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                 frames=range(0, n_frames), interval=300, repeat=True)
        plt.show()
        return 0

    vector_difference = [x[1]-x[0], y[1]-y[0], z[1]-z[0]]
    mag_vector_difference = math.sqrt(
        vector_difference[0]**2 + vector_difference[1]**2 + vector_difference[2]**2)
    n_actuations = math.ceil(mag_vector_difference/link_length)
    theta = theta_calc(
        (vector_difference[0], vector_difference[1], vector_difference[2]), (0, 1, 0))
    phi = phi_calc(
        (vector_difference[0], vector_difference[1], vector_difference[2]), (-1, 0, 0))

    if phi > curvature_limit:
        bezier_main(0, 0, 0, actuations, x_animation_array, y_animation_array,
                    z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array)
    else:
        # Adjusting for range of theta_calc being [0, pi]
        if vector_difference[0] > 0 and vector_difference[1] < 0:
            theta = theta + math.pi
        elif vector_difference[0] < 0 and vector_difference[1] < 0:
            theta = theta + math.pi
        # Calculate unit vector of vector_difference
        for i in range(0, 3):
            vector_difference[i] = vector_difference[i] / mag_vector_difference

        # Update current position
        current_pos = [current_pos[0] + n_actuations*vector_difference[0]*link_length, current_pos[1] + n_actuations *
                       vector_difference[1]*link_length, current_pos[2] + n_actuations*vector_difference[2]*link_length]

        # First, append current theta and phi. Then append 0 to represent snake continuing to move along the line
        phi_array.append(phi)
        theta_array.append(theta)
        for j in range(0, n_actuations-1):
            phi_array.append(0)
            theta_array.append(0)
        x[0] = current_pos[0]
        x.pop(1)
        y[0] = current_pos[1]
        y.pop(1)
        z[0] = current_pos[2]
        z.pop(1)

        classic_follow(x, y, z, current_pos, x_animation_array, y_animation_array,
                       z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array, theta_array, phi_array)


def bezier_main(xBezier, yBezier, zBezier, actuations, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array):
    current_pos = [x[0], y[0], z[0]]
    path_length = 0
    k = 0
    previous_vector = [0, 0, 0]
    curvature_limit = math.pi/6
    phi = 0
    actuations = actuations
    xBezier, yBezier, zBezier = first_bezier(x, y, z)

    def recursive_bezier(path_length, k, xBezier, yBezier, zBezier, previous_vector, current_pos, phi, actuations, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array):
        if k == len(xBezier):
            print("End of curve reached")
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111, projection='3d')

            ax1.plot(
                x_classic, y_classic, z_classic, c='green')
            ax1.set_xlabel("x axis")
            ax1.set_ylabel("y axis")
            ax1.set_zlabel("z axis")

            def animate(i):
                ax1.cla()
                ax1.scatter(
                    x_animation_array[i], y_animation_array[i], z_animation_array[i], c='black')
                current_pos_x = []
                current_pos_y = []
                current_pos_z = []
                for j in range(0, i):
                    current_pos_x.append(current_pos_array[j][0])
                    current_pos_y.append(current_pos_array[j][1])
                    current_pos_z.append(current_pos_array[j][2])
                ax1.plot(
                    current_pos_x, current_pos_y, current_pos_z, c='blue', linewidth=5)
                ax1.plot(
                    xBezier_array[i], yBezier_array[i], zBezier_array[i], c='red')
                fig.canvas.draw_idle()

            n_frames = len(xBezier_array)
            ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                     frames=range(0, n_frames), interval=300, repeat=True)
            plt.show()
            return 0
        elif actuations > max_actuations:
            print("Snake fully extended")
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(111, projection='3d')

            ax1.plot(
                x_classic, y_classic, z_classic, c='green')
            ax1.set_xlabel("x axis")
            ax1.set_ylabel("y axis")
            ax1.set_zlabel("z axis")

            def animate(i):
                ax1.cla()
                ax1.scatter(
                    x_animation_array[i], y_animation_array[i], z_animation_array[i], c='black')
                current_pos_x = []
                current_pos_y = []
                current_pos_z = []
                for j in range(0, i):
                    current_pos_x.append(current_pos_array[j][0])
                    current_pos_y.append(current_pos_array[j][1])
                    current_pos_z.append(current_pos_array[j][2])
                ax1.plot(
                    current_pos_x, current_pos_y, current_pos_z, c='blue', linewidth=5)
                ax1.plot(
                    xBezier_array[i], yBezier_array[i], zBezier_array[i], c='red')
                fig.canvas.draw_idle()

            n_frames = len(xBezier_array)
            ani = matplotlib.animation.FuncAnimation(fig, animate,
                                                     frames=range(0, n_frames), interval=300, repeat=True)
            plt.show()
            return 0
        path_length = math.sqrt((xBezier[k]-current_pos[0])**2 + (
            yBezier[k]-current_pos[1])**2 + (zBezier[k] - current_pos[2])**2)

        if path_length > link_length:
            vector = [xBezier[k] - current_pos[0], yBezier[k] -
                      current_pos[1], zBezier[k] - current_pos[2]]
            mag_vector = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
            un_vector = [vector[0]/mag_vector,
                         vector[1]/mag_vector, vector[2]/mag_vector]
            current_pos = [current_pos[0] + un_vector[0]*link_length, current_pos[1] +
                           un_vector[1]*link_length, current_pos[2] + un_vector[2]*link_length]

            # Theta and phi calculation
            theta = theta_calc(vector, previous_vector)
            if vector[0] - previous_vector[0] < 0 and vector[1] - previous_vector[1] < 0:
                theta = theta + math.pi
            elif vector[0] - previous_vector[0] > 0 and vector[1] - previous_vector[1] < 0:
                theta = theta + math.pi
            phi_prev = phi
            phi = phi_calc(vector, previous_vector)
            if abs(phi_prev-phi) > curvature_limit:
                request_control_point_add = (current_pos[0] + link_length*math.cos(curvature_limit),
                                             current_pos[1] + link_length*math.sin(curvature_limit)*math.cos(theta), current_pos[2] + link_length*math.sin(curvature_limit)*math.sin(theta))
                xBezier, yBezier, zBezier = new_bezier(
                    x, y, z, xBezier, yBezier, zBezier, request_control_point_add, current_pos)
                previous_vector = [0, 0, 0]
            else:
                previous_vector = vector
                actuations += 1
                phi_array.append(phi)
                theta_array.append(theta)
                path_length = 0
            xBezier_array.append(xBezier)
            yBezier_array.append(yBezier)
            zBezier_array.append(zBezier)
            current_pos_array.append(current_pos)
            x_animation_array.append(x.copy())
            y_animation_array.append(y.copy())
            z_animation_array.append(z.copy())
            recursive_bezier(path_length, k, xBezier, yBezier, zBezier,
                             previous_vector, current_pos, phi, actuations, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array)

        else:
            k += 1
            recursive_bezier(path_length, k, xBezier, yBezier, zBezier,
                             previous_vector, current_pos, phi, actuations, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array)

    recursive_bezier(path_length, k, xBezier, yBezier, zBezier,
                     previous_vector, current_pos, phi, actuations, x_animation_array, y_animation_array, z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array)


def first_bezier(x, y, z):
    cells = 1000

    n_control_points = len(x)

    n = n_control_points - 1
    i = 0
    t = np.linspace(0, 1, cells)

    b = []

    xBezier = np.zeros((1, cells))
    yBezier = np.zeros((1, cells))
    zBezier = np.zeros((1, cells))

    def Ni(n, i):
        return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n-i))

    def basisFunction(n, i, t):
        J = np.array(Ni(n, i) * (t**i) * (1-t) ** (n-i))
        return J

    for k in range(0, n_control_points):
        b.append(basisFunction(n, i, t))

        xBezier = basisFunction(n, i, t) * x[k] + xBezier
        yBezier = basisFunction(n, i, t) * y[k] + yBezier
        zBezier = basisFunction(n, i, t) * z[k] + zBezier

        i += 1

    return xBezier[0], yBezier[0], zBezier[0]


def new_bezier(x, y, z, xPreviousBezier, yPreviousBezier, zPreviousBezier, requested_goal, current_pos):
    closest_k = []

    minimum_length = math.sqrt(xPreviousBezier[len(xPreviousBezier)-1]**2 + yPreviousBezier[len(
        yPreviousBezier)-1]**2 + zPreviousBezier[len(zPreviousBezier)-1]**2)

    minimum_j = 0

    # Find relative positions of controls points to previous bezier
    for i in range(0, len(x)):
        current_goal = [x[i], y[i], z[i]]
        minimum_k = 0
        for k in range(0, len(xPreviousBezier)):
            current_length = math.sqrt((xPreviousBezier[k]-current_goal[0])**2 + (
                yPreviousBezier[k]-current_goal[1])**2 + (zPreviousBezier[k]-current_goal[2])**2)
            if current_length <= minimum_length:
                minimum_length = current_length
                minimum_k = k
        closest_k.append(minimum_k)

    # Find closest point to added control point on previous bezier
    for j in range(0, len(xPreviousBezier)):
        current_length = math.sqrt((xPreviousBezier[j]-requested_goal[0])**2 + (
            yPreviousBezier[j]-requested_goal[1])**2 + (zPreviousBezier[j]-requested_goal[2])**2)
        if current_length < minimum_length:
            minimum_length = current_length
            minimum_j = j

    for b in range(0, len(closest_k)):
        if minimum_j < closest_k[b]:
            x.insert(b, requested_goal[0])
            y.insert(b, requested_goal[1])
            z.insert(b, requested_goal[2])

    cells = len(xPreviousBezier)

    n_control_points = len(x)

    n = n_control_points - 1
    i = 0
    t = np.linspace(0, 1, cells)

    b = []

    xBezier = np.zeros((1, cells))
    yBezier = np.zeros((1, cells))
    zBezier = np.zeros((1, cells))

    def Ni(n, i):
        return np.math.factorial(n) / (np.math.factorial(i) * np.math.factorial(n-i))

    def basisFunction(n, i, t):
        J = np.array(Ni(n, i) * (t**i) * (1-t) ** (n-i))
        return J

    for k in range(0, n_control_points):
        b.append(basisFunction(n, i, t))

        xBezier = basisFunction(n, i, t) * x[k] + xBezier
        yBezier = basisFunction(n, i, t) * y[k] + yBezier
        zBezier = basisFunction(n, i, t) * z[k] + zBezier

        i += 1

    return xBezier[0], yBezier[0], zBezier[0]


classic_follow(x, y, z, current_pos, x_animation_array, y_animation_array,
               z_animation_array, xBezier_array, yBezier_array, zBezier_array, current_pos_array, theta_array, phi_array)
print(phi_array)
print(theta_array)
