#!/usr/bin/env python
# coding: utf-8


# Combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
import cv2
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndarray
from skimage import transform
from skimage import util

img = cv2.imread('1.png')


def random_crop(img):
    h = random.randint(1, ((img.shape)[0]))
    w = random.randint(1, ((img.shape)[1]))
    img_crop = img[0:h, 0:w]
    print("start------1.frist step: crop image. For next step press 'Enter'------")
    cv2.imshow('crop_image', img_crop)
    cv2.waitKey()
    #   cv2.destroyAllWindows()
    return img_crop


def random_color_shift(img_crop):
    B, G, R = cv2.split(img_crop)

    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)

    img_merge = cv2.merge((B, G, R))
    print("------2.second step: shift color.  For next step press 'Enter'------")
    cv2.imshow('shift_color_image', img_merge)
    cv2.waitKey()
    #     cv2.destroyAllWindows()
    return img_merge


def random_rotation(img_merge):
    random_angle = random.uniform(-90, 90)
    M = cv2.getRotationMatrix2D((img_merge.shape[1] / 2, img_merge.shape[0] / 2), random_angle,
                                1)  # center, angle, scale
    img_rotate = cv2.warpAffine(img_merge, M, (img_merge.shape[1], img_merge.shape[0]))
    print("------3.third step: rotate image.  For next step press 'Enter'------")
    cv2.imshow('rotate_image', img_rotate)
    cv2.waitKey()
    #     cv2.destroyAllWindows()
    return img_rotate


def random_warp(img):
    height = (img.shape)[0]
    width = (img.shape)[1]

    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype="float32")
    pts2 = np.array([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]], dtype="float32")

    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))

    print("------4.fourth step: perspective transform. ------End ")
    cv2.imshow('perspective_transform', img_warp)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return img_warp


random_crop_img = random_crop(img)
random_color_shift_img = random_color_shift(random_crop_img)
random_rotation_img = random_rotation(random_color_shift_img)
finised_img = random_warp(random_rotation_img)