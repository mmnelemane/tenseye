# Bruteforce Matching

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/feature.jpg', 0)
template = cv2.imread('images/feature_template.jpg', 0)

orb = cv2.ORB_create()

kp_t, des_t = orb.detectAndCompute(template, None)
kp_i, des_i = orb.detectAndCompute(img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des_t, des_i)
matches = sorted(matches, key = lambda x:x.distance)

res = cv2.drawMatches(template, kp_t, img, kp_i, matches[:30], None, flags=2)
plt.imshow(res)
plt.show()

