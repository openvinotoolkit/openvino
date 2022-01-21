# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import copy

import cv2
import numpy as np


def generate_bounding_box(mapping, reg, scale, threshold):
    stride = 2
    cellsize = 12
    mapping = mapping.T
    (x, y) = np.where(mapping >= threshold)
    reg = np.array([reg[0].T[x, y], reg[1].T[x, y], reg[2].T[x, y], reg[3].T[x, y]])

    bounding_box = np.array([y, x]).T
    bb1 = np.fix((stride * bounding_box + 1) / scale).T
    bb2 = np.fix((stride * bounding_box + cellsize - 1 + 1) / scale).T

    score = np.array([mapping[x, y]])

    return np.row_stack((bb1, bb2, score, reg)).T


def nms(boxes, threshold, overlap_type='union'):
    if boxes.shape[0] == 0:
        return np.array([])
    x_min, x_max = boxes[:, 0], boxes[:, 2]
    y_min, y_max = boxes[:, 1], boxes[:, 3]
    area = np.multiply(x_max - x_min + 1, y_max - y_min + 1)
    idxs = boxes[:, 4].argsort()

    pick = []
    while np.size(idxs) > 0:
        pick.append(idxs[-1])
        xx_min = np.maximum(x_min[idxs[-1]], x_min[idxs[0:-1]])
        xx_max = np.minimum(x_max[idxs[-1]], x_max[idxs[0:-1]])
        yy_min = np.maximum(y_min[idxs[-1]], y_min[idxs[0:-1]])
        yy_max = np.minimum(y_max[idxs[-1]], y_max[idxs[0:-1]])
        width = np.clip(xx_max - xx_min + 1, 0.0, None)
        height = np.clip(yy_max - yy_min + 1, 0.0, None)
        intersection = width * height
        iou = (intersection / np.minimum(area[idxs[-1]], area[idxs[0:-1]])) if overlap_type == 'min' \
            else (intersection / (area[idxs[-1]] + area[idxs[0:-1]] - intersection))
        idxs = idxs[np.flatnonzero(iou <= threshold)]

    return pick


def cut_roi(image, bboxes, dst_size, include_bound=True):
    bboxes = reshape_to_square(copy(bboxes))
    bboxes[:, 0:4] = np.fix(bboxes[:, 0:4])
    roi_samples = []
    for dy, edy, dx, edx, y, ey, x, ex, roi_w, roi_h in zip(*pad(bboxes, *image.shape[:2])):
        roi_h = roi_h + int(include_bound)
        roi_w = roi_w + int(include_bound)
        roi = np.zeros((roi_h, roi_w, 3))
        roi[dy:edy + 1, dx:edx + 1] = image[y:ey + 1, x:ex + 1]
        roi_samples.append(cv2.resize(roi, (dst_size, dst_size)))
    return np.array(roi_samples)


def reshape_to_square(bbox):
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    dst_length = np.maximum(w, h)
    hw_correction = np.transpose([w * 0.5 - dst_length * 0.5, h * 0.5 - dst_length * 0.5])

    bbox[:, 0:2] += hw_correction
    bbox[:, 2:4] = bbox[:, 0:2] + np.repeat([dst_length], 2, axis=0).T

    return bbox


def pad(boxes, h, w):

    tmph = boxes[:, 3] - boxes[:, 1] + 1
    tmpw = boxes[:, 2] - boxes[:, 0] + 1

    dx, dy = np.ones(len(boxes)), np.ones(len(boxes))
    x, y, ex, ey = boxes[:, 0:4].T

    tmp = np.flatnonzero(ex > w)
    if len(tmp) != 0:
        tmpw[tmp] = -ex[tmp] + w - 1 + tmpw[tmp]
        ex[tmp] = w - 1

    tmp = np.flatnonzero(ey > h)
    if len(tmp) != 0:
        tmph[tmp] = -ey[tmp] + h - 1 + tmph[tmp]
        ey[tmp] = h - 1

    tmp = np.flatnonzero(x < 1)
    if len(tmp) != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.flatnonzero(y < 1)
    if len(tmp) != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])

    dy, dx, y, x = np.maximum(np.zeros((4, len(boxes))), [dy - 1, dx - 1, y - 1, x - 1]).astype(int)
    edy, edx, ey, ex = np.maximum(np.zeros((4, len(boxes))), [tmph - 1, tmpw - 1, ey - 1, ex - 1]).astype(int)

    return dy, edy, dx, edx, y, ey, x, ex, tmpw.astype(int), tmph.astype(int)


def bbreg(bbox, reg, include_bound=True):
    width = bbox[:, 2] - bbox[:, 0] + int(include_bound)
    height = bbox[:, 3] - bbox[:, 1] + + int(include_bound)

    bbox[:, 0] += reg[:, 0] * width
    bbox[:, 1] += reg[:, 1] * height
    bbox[:, 2] += reg[:, 2] * width
    bbox[:, 3] += reg[:, 3] * height

    return bbox


def calibrate_bboxes(bboxes, scores, regions, nms_type=None):
    pass_t = np.flatnonzero(scores > 0.7)
    bboxes_to_remove = np.setdiff1d(list(range(len(bboxes))), pass_t)
    filtered_bboxes = np.delete(bboxes, bboxes_to_remove, axis=0)
    filtered_bboxes[:, 4] = scores[pass_t]
    mv = regions[pass_t]

    if nms_type:
        pick = nms(filtered_bboxes, 0.7, nms_type)
        bboxes_to_remove = np.setdiff1d(np.arange(len(filtered_bboxes)), pick)
        filtered_bboxes = np.delete(filtered_bboxes, bboxes_to_remove, axis=0)
        mv = mv[np.sort(pick).astype(int)]

    return bbreg(filtered_bboxes, mv)


def build_image_pyramid(image, scale_factor, m, layout='NCHW'):
    img = image.astype(float)
    height, width, _ = img.shape
    min_layer = min(height, width) * m
    scales, image_pyramid = [], []
    factor_count = 0
    while min_layer >= 12:
        scale = m * pow(scale_factor, factor_count)
        scales.append(scale)
        height_scaled = int(np.ceil(height * scale))
        width_scaled = int(np.ceil(width * scale))
        scaled_image = cv2.resize(img, (width_scaled, height_scaled))
        image_pyramid.append(np.transpose([scaled_image], [0, 3, 2, 1])
                             if layout == 'NCHW' else scaled_image)
        min_layer *= scale_factor
        factor_count += 1
    return image_pyramid, scales


def calculate_tp(prediction, annotation, overlap_threshold=0.5):
    annotation_boxes = np.array(annotation['bboxes'])
    num_recorded_gt = len(annotation_boxes)
    difficult_box_mask = np.full(num_recorded_gt, False)
    difficult_box_mask[annotation['difficult']] = True
    num_recorded_gt -= len(annotation['difficult'])
    score_order = np.argsort(-prediction[:, 4])
    prediction = prediction[score_order]

    tp = np.zeros(len(prediction))
    used = np.zeros(len(annotation_boxes))
    for image, _ in enumerate(prediction):

        prediction_box = prediction[image][:-1]

        iou = calculate_iou(prediction_box, annotation_boxes)
        ignored = annotation['difficult']
        ignored_annotation_boxes = annotation_boxes[ignored]
        iou[ignored] = calculate_iou(prediction_box, ignored_annotation_boxes, over_area=True)

        max_iou = -np.inf

        not_ignored_overlaps = iou[~difficult_box_mask]
        ignored_overlaps = iou[difficult_box_mask]
        if not_ignored_overlaps.size:
            max_iou = np.max(not_ignored_overlaps)

        if max_iou < overlap_threshold and ignored_overlaps.size:
            max_iou = np.max(ignored_overlaps)
        max_overlapped = np.flatnonzero(iou == max_iou)

        if max_iou < overlap_threshold:
            continue
        if not difficult_box_mask[max_overlapped].any() and not used[max_overlapped].any():
            used[max_overlapped] = True
            tp[image] = 1

    return tp, num_recorded_gt


#pylint: disable=E1136
def calculate_iou(prediction, annotation, over_area=False):
    p_tmp = np.full_like(annotation, prediction, dtype=prediction.dtype).T
    inter_x_min, inter_y_min = np.maximum(annotation.T[:2], p_tmp[:2])
    inter_x_max, inter_y_max = np.minimum(annotation.T[2:], p_tmp[2:])
    inter_x_max = np.maximum(inter_x_min, inter_x_max)
    inter_y_max = np.maximum(inter_y_min, inter_y_max)

    def area(x, y, xx, yy):
        return (xx - x + 1) * (yy - y + 1)

    intersection = area(inter_x_min, inter_y_min, inter_x_max, inter_y_max)
    union = (area(*prediction) + area(*annotation.T) - intersection) if not over_area else area(*prediction)
    return np.divide(intersection, union,
                     out=np.zeros_like(intersection, dtype=float), where=union != 0)
