###############################################################################
# 1. Importing libraries
###############################################################################
# import tensorflow
# import keras
# from tensorflow import keras
import os
import glob
from skimage import io
# import random
# import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

###############################################################################
# 2. Importing the Dataset
###############################################################################

# Importing and Loading the data into a data frame
# dataset_path = '/content/drive/MyDrive/Animals'
# class_names = ['Cheetah', 'Jaguar', 'Leopard', 'Lion','Tiger']

current_dir = os.getcwd()
relative_path = "Test_Images/video9"

# apply glob module to retrieve files/pathnames
# animal_path = os.path.join(dataset_path, class_names[1], '*')
# animal_path = glob.glob(animal_path)

test_images_folder_path = os.path.join(current_dir, relative_path)
# print(test_images_folder_path)
# test_images_path = glob.glob(test_images_path)
# print(test_images_path)

test_images = []
for image in os.listdir(test_images_folder_path):
    image_path = os.path.join(test_images_folder_path, image)
    test_images.append(image_path)
# print(test_images)
print(test_images[4])

# accessing an image file from the dataset classes
# image = io.imread(animal_path[4])
image = io.imread(test_images[4])

# plotting the original image !!! ONLY WORKS IN NOTEBOOKS
i, (im1) = plt.subplots(1)
i.set_figwidth(15)
im1.imshow(image)

###############################################################################
# 3. Data preprocessing
###############################################################################

# 3.1. Plotting the original image and the RGB channels
i, (im1, im2, im3, im4) = plt.subplots(1, 4, sharey=True)
i.set_figwidth(20)

im1.imshow(image)  #Original image
im2.imshow(image[:, : , 0]) #Red
im3.imshow(image[:, : , 1]) #Green
im4.imshow(image[:, : , 2]) #Blue
i.suptitle('Original & RGB image channels')


# # 3.2. Grayscale conversion
# gray_image = skimage.color.rgb2gray(image)
# plt.imshow(gray_image, cmap = 'gray')

# # 3.3. Normalization
# norm_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))
# plt.imshow(norm_image)

# # 3.4. Data augmentation

# # 3.4.1. Shifting: This is the process of shifting image pixels horizontally or vertically.

# # import libraries
# from numpy import expand_dims
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator

# # convert to numpy array
# data = img_to_array(image)

# # expand dimension to one sample
# samples = expand_dims(image, 0)

# # create image data augmentation generator
# datagen = ImageDataGenerator(width_shift_range=[-200,200])

# # create an iterator
# it = datagen.flow(samples, batch_size=1)
# fig, im = plt.subplots(nrows=1, ncols=3, figsize=(15,15))

# # generate batch of images
# for i in range(3):

#     # convert to unsigned integers
#     image = next(it)[0].astype('uint8')

#     # plot image
#     im[i].imshow(image)

# # 3.4.2. Flipping: This reverses the rows or columns of pixels in either vertical or horizontal cases, respectively.
# # ImageDataGenerator for flipping
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


# # 3.4.4. Rotation: This process involves rotating an image by a specified degree.
# datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')

# # 3.4.5 Changing brightness: This is the process of increasing or decreasing image contrast.
# datagen = ImageDataGenerator(brightness_range=[0.5,2.0])


# # 3.4.6. Standardizing images: Standardization is a method that scales and
# # preprocesses images to have similar heights and widths. It re-scales data to
# # have a standard deviation of 1 (unit variance) and a mean of 0.
# # creating the image data generator to standardize images
# datagen = ImageDataGenerator(featurewise_center =True,
#       featurewise_std_normalization = True)
