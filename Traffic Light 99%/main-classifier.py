import random

import helpers
import masks
import test_functions


# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"


# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


# Test for one_hot_encode function
tests = test_functions.Tests()
tests.test_one_hot(helpers.one_hot_encode)


# Standardize all training images
STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)


#This function takes an rgb image, xtract feature(s) from it and uses those
#features to classify the image and output a one-hot encoded label.
def estimate_label(rgb_image):
    predicted_label = []

    y_count = masks.yellow(rgb_image)
    r_count = masks.red(rgb_image)
    g_count = masks.green(rgb_image)

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
STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)


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
helpers.display_image(MISCLASSIFIED[0])


if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
