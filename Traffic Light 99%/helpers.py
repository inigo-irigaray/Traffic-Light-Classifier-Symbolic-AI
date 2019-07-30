import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2 #computer vision library


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    # Iterate through each color folder
    for im_type in image_types:
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):

            # Read in the image
            im = mpimg.imread(file)

            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))
    return im_list

#This function takes an image and displays it
def display_image(image):
    plt.imshow(image[0])
    plt.show()
    print("Shape: " + str(image[0].shape))
    print("Label: " + str(image[1]))


# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    #Resize image and pre-process so that all "standard" images are the same size
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    return standard_im


# Given a label - "red", "green", or "yellow" - return a one-hot encoded label
def one_hot_encode(label):
    #Create a one-hot encoded label that works for all classes of traffic lights
    one_hot_encoded = []
    if label == 'red':
        one_hot_encoded = [1, 0, 0]
    elif label == 'yellow':
        one_hot_encoded = [0, 1, 0]
    else:
        one_hot_encoded = [0, 0, 1]
    return one_hot_encoded


#This function takes in an image_list, stadardizes and one-hot encodes the label of
#each image in the list.
def standardize(image_list):
    # Empty image data array
    standard_list = []
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]
        # Standardize the image
        standardized_im = standardize_input(image)
        # One-hot encode the label
        one_hot_label = one_hot_encode(label)
        # Append the image, and it's one hot encoded label to the full, processed list of image data
        standard_list.append((standardized_im, one_hot_label))
    return standard_list
