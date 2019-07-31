import numpy as np
import cv2

#This function applies a saturation mas to the image
def saturation(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    s = hsv[:,:, 1]

    height, width, color = hsv.shape

    lower_bound = (np.sum(s) / (height * width)) + np.std(s)
    upper_bound = 256

    mask = cv2.inRange(s, lower_bound, upper_bound)

    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = 0

    ###f, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

    ###ax1.set_title('Saturation sum over rows')
    ###ax1.plot(np.sum(s, axis=1))

    ###ax2.set_title('Masked image')
    ###ax2.imshow(masked_image)
    return masked_image

#This function applies a brightness mask to the image
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

    ###vSUM = np.sum(masked_image, axis = 1)
    ###f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ###ax1.set_title('Value sum over rows')
    ###ax1.plot(vSUM)

    ###ax2.set_title('Masked image')
    ###ax2.imshow(masked_image)

    return masked_image

#This function counts red pixels
def red(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([107, 0, 0])
    upper_bound = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    ###plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1

    return count

#This function counts yellow pixels
def yellow(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([15, 0, 0])
    upper_bound = np.array([70, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    ###plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1
    return count

#This function counts green pixels
def green(rgb_image):
    image = brightness(rgb_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([70, 0, 0])
    upper_bound = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    masked_image = np.copy(image)
    masked_image[mask==0] = 0

    ###plt.imshow(masked_image)

    count = 0

    height, width, color = masked_image.shape
    for i in range(height):
        for j in range(width):
            for z in range(color):
                if masked_image[i][j][z] != 0:
                    count += 1
    return count
