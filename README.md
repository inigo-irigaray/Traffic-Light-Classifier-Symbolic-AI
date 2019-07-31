# Traffic-Light-Classifier-Symbolic-AI

This project aims to build a fully functional symbolic AI traffic light classifier, using a variety of computer vision techniques, to be integrated into a self-driving car. It successfully classified 97% of the images, with no red misclassifications as green to maximize safety. A second version of the code achieves 99% accuracy, but with one red misclassification as green of a low-quality image.


This program pre-processes the images, extracts features that will help distinguish the different types of images, and uses these features to classify the traffic light images into classes corresponding to each the light color.

**1. Loading and visualizing the data.** The first step was developing a display function to visualize and familiarize myself with the data.

**2. Pre-processing.** Secondly, the images and output labels needed to be standardized. Thus, all the images can be analyzed using the same classification pipeline, and we know what output to expect for every new image classification.

**3. Feature extraction.** Thirdly, I applied some filters and masks to extract the features from the images that will help to classify them into the three categories.

**4. Classification and visualizing the error.** Finally, I wrote the function that integrates image pre-processing, feature extraction and visualization to classify any traffic light. It takes an image as an input and outputs a label, and determines the model's accuracy.

---------

# 1. Prerequisites.

- OpenCV
- NumPy
- MatPlotLib
- IPython

# 2. Loading and visualizing the data.

This traffic light dataset consists of 1484 number of color images in 3 categories - red, yellow, and green. As with most human-sourced data, the data is not evenly distributed among the types. There are:

* 904 red traffic light images
* 536 green traffic light images
* 44 yellow traffic light images

Note: All images come from this [MIT self-driving car course](https://selfdrivingcars.mit.edu/) and are licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
<p float="left">
  
<img src="https://github.com/inigo-irigaray/Traffic-Light-Classifier-Symbolic-AI/blob/master/Traffic-Light-99%25/images/Green-Light.png" height = 300 width=250><img src="https://github.com/inigo-irigaray/Traffic-Light-Classifier-Symbolic-AI/blob/master/Traffic-Light-99%25/images/Red-Light.png" height = 300 width=250><img src="https://github.com/inigo-irigaray/Traffic-Light-Classifier-Symbolic-AI/blob/master/Traffic-Light-99%25/images/Yellow-Light.png" height = 300 width=270>
  
</p>

# 3. Pre-processing.

For classification tasks like this one we need to create features by performing the same analysis on different pictures. It is, therefore, important that similar images create similar features. And to facilitate this, we will standardize the input and output to understand what results we can expect from running the program.

First, I created a function that takes an image and resizes it to 32x32. Square images can be rotated and analyzed in smaller square patches. Additionally, if all images are the same size we can confidently pass them through the same pipeline.

Secondly, it is customary to convert categorical labels like 'red' to numerical values. I created a function that one-hot encodes the labels into a 1D list of zeros with a number one representing the categorical value in the following way:
  - red     =   [1,0,0]
  - yellow  =   [0,1,0]
  - green   =   [0,0,1]

Finally, I created a function to standardize a list of images and pair each image to its one-hot encoded label.

# 4. Feature extraction.

