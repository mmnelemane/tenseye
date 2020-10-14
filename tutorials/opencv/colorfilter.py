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

    kernel = np.ones((15,15), np.float32)/225
    smoothed = cv2.filter2D(res, -1, kernel)

    blur = cv2.GaussianBlur(res, (15,15), 0)
    median = cv2.medianBlur(res, 15)
    bilateral = cv2.bilateralFilter(res, 15, 75, 75)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', res)
    cv2.imshow('smoothed', smoothed)
    cv2.imshow('blurred', blur)
    cv2.imshow('median', median)
    cv2.imshow('bilateral', bilateral)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

