"""
 Copyright (C) 2018-2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import cv2
import numpy as np


def compare_images(img_path1, img_path2, eps):
    assert os.path.isfile(img_path1), "Image after infer was not found: {}".format(img_path1)
    assert os.path.isfile(img_path2), "Reference image was not found: {}".format(img_path2)
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    acc_pass = np.allclose(img1, img2, atol=eps, rtol=eps)
    return acc_pass


def check_image_if_box(img_path1, img_path2):
    # Case with batch,
    if ' ' in img_path2:
        img_path2 = img_path2.split(' ')[0]
    assert os.path.isfile(img_path1), "Image after infer was not found: {}".format(img_path1)
    assert os.path.isfile(img_path2), "Reference image was not found: {}".format(img_path2)
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    absdiff = cv2.absdiff(img1, img2)
    acc_pass = absdiff.sum()
    return acc_pass


def iou(curr_rectangle, reff_rectangle):
    # determine the (x, y)-coordinates of the intersection rectangle
    xCurr = max(curr_rectangle[0], reff_rectangle[0])
    yCurr = max(curr_rectangle[1], reff_rectangle[1])
    xReff = min(curr_rectangle[2], reff_rectangle[2])
    yReff = min(curr_rectangle[3], reff_rectangle[3])

    # compute the area of intersection rectangle
    interArea = (xReff - xCurr + 1) * (yReff - yCurr + 1)

    # compute the area of both the current and refference rectangles
    boxCurrentArea = (curr_rectangle[2] - curr_rectangle[0] + 1) * (curr_rectangle[3]
                                                                    - curr_rectangle[1] + 1)
    boxRefferenceArea = (reff_rectangle[2] - reff_rectangle[0] + 1) * (reff_rectangle[3]
                                                                       - reff_rectangle[1] + 1)
    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # current + refference areas - the interesection area
    iou = interArea / float(boxCurrentArea + boxRefferenceArea - interArea)
    return iou
