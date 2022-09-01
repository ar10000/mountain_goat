import cv2
import os, sys
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# path to image
coordinates = [[0.34432, 0.34222, 1],
               [0.1, 0.8724, 0.9855],
               [0.3426, 0.2312, 0.3214],
               [0.2431, 0.2342, 0.6958]]

def plot_coordinates(coordinates):
    c_t = np.transpose(coordinates)
    plt.rcParams["figure.figsize"] = [14.00, 7]
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #data = np.random.random(size=(3, 3, 3))
    z, x, y = c_t[2], c_t[0], c_t[1]
    ax.scatter(x, y, z, c=z, alpha=1)

    #plt.annotate(xy = (c_t[0][0],c_t[0][1]), text = "Hello")
    #plt.annotate(x = c_t[1][0], y=y, text = "Hello")
    #plt.annotate(x = c_t[1][0], y=y, text = "Hello")
    #plt.annotate(x = c_t[0][0], y=y, text = "Hello")

    plt.show()

plot_coordinates(coordinates)
