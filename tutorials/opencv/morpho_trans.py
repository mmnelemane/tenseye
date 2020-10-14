# Color filtering, smooting and blurring
# Morphological Transformations for removing noise

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # hsv = Hue Saturation Value
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 0, 210])
    upper_blue = np.array([255, 255, 255])

    # dark_green = np.uint8([[[12, 22, 121]]])
    # dark_green = cv2.cvtColor(dark_green, cv2.COLOR_BGR2HSV)

    # mask helps in color filtering
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Erosion
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = 1)
    dilation = cv2.dilate(mask, kernel, iterations = 1)

    #opening removes false positives
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # It is the difference between input image and Opening of the image
    # cv2.imshow('Tophat',tophat)

    # It is the difference between the closing of the input image and input image.
    # cv2.imshow('Blackhat',blackhat)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)
    cv2.imshow('erosion', erosion)
    cv2.imshow('dilation', dilation)
    cv2.imshow('opening', opening)
    cv2.imshow('closing', closing)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

