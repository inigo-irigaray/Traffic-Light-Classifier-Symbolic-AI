import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 #computer vision library

import helpers # helper functions
import test_functions

# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

def display_image(image):
    plt.imshow(image[0])
    plt.show()
    print("Shape: " + str(image[0].shape))
    print("Label: " + str(image[1]))


#THE FOLLOWING LINES TEST THE DISPLAY FUNCTION.
for i in range(len(IMAGE_LIST)):
    if IMAGE_LIST[i][1] == 'red':
        print('Image: ' + str(i))
        display_image(IMAGE_LIST[i])
        break

for i in range(len(IMAGE_LIST)):
    if IMAGE_LIST[i][1] == 'yellow':
        print('Image: ' + str(i))
        display_image(IMAGE_LIST[i])
        break

for i in range(len(IMAGE_LIST)):
    if IMAGE_LIST[i][1] == 'green':
        print('Image: ' + str(i))
        display_image(IMAGE_LIST[i])
        break



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


# Test for one_hot_encode function
tests = test_functions.Tests()
tests.test_one_hot(one_hot_encode)


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

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)


def saturation(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    s = hsv[:,:, 1]

    height, width, color = hsv.shape

    lower_bound = (np.sum(s) / (height * width)) + np.std(s)
    upper_bound = 256

    mask = cv2.inRange(s, lower_bound, upper_bound)

    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = 0

    #f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

    #ax1.set_title('Saturation sum over rows')
    #ax1.plot(np.sum(s, axis=1))

    #ax2.set_title('Masked image')
    #ax2.imshow(masked_image)
    return masked_image

def brightness(rgb_image):
    image = saturation(rgb_image)

    #Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:,:, 2] #displays image brightness

    v_sum = np.sum(v)
    v_std = np.std(v) * 2
    area = 32 * 32

    lower_bound1 = (v_sum / area) + v_std
    upper_bound1 = 256

    mask1 = cv2.inRange(v, lower_bound1, upper_bound1) #selects sections brightest sections of h

    masked_image = np.copy(image)
    masked_image[mask1 == 0] = 0 #assigns maximum darkness to sections outside the brightest section

    #vSUM = np.sum(masked_image, axis = 1)
    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    #ax1.set_title('Value sum over rows')
    #ax1.plot(vSUM)

    #ax2.set_title('Masked image')
    #ax2.imshow(masked_image)

    return masked_image

def red(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([107, 0, 0])
    upper_bound = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    #plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1

    return count

def yellow(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([15, 0, 0])
    upper_bound = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    #plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1
    return count

def green(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([70, 0, 0])
    upper_bound = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    #plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1
    return count



def estimate_label(rgb_image):

    #Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = []

    y_count = yellow(rgb_image)
    r_count = red(rgb_image)
    g_count = green(rgb_image)

    if r_count > y_count and r_count > g_count:
        predicted_label = [1,0,0]
    elif y_count > r_count and y_count > g_count:
        predicted_label = [0,1,0]
    else:
        predicted_label = [0, 0, 1]

    return predicted_label




# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)


# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

## Visualize misclassified example(s)
#Display an image in the `MISCLASSIFIED` list
#Print out its predicted label - to see what the image *was* incorrectly classified as
display_image(MISCLASSIFIED[6])


if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
