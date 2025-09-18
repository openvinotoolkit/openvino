#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import numpy as np


def compare_blobs(cpu_filename, kmb_filename):
    cpu_img = np.fromfile(cpu_filename, np.float16)
    cpu_img = cpu_img.astype(np.float16)
    cpu_img = cpu_img.astype(np.float32)
    npu_img = np.fromfile(kmb_filename, np.float32)
    if len(npu_img) == len(cpu_img) // 2:
        npu_img = np.fromfile(kmb_filename, np.float16)
        npu_img = npu_img.astype(np.float32)

    cpu_img_size = cpu_img.nbytes
    npu_img_size = npu_img.nbytes

    if cpu_img_size != npu_img_size:
        raise RuntimeError(f"Blobs are not alligned, sizes: {cpu_img_size}/{npu_img_size}")

    npu_img[np.isnan(npu_img)] = 0
    npu_img[np.isneginf(npu_img)] = -65000
    npu_img[np.isinf(npu_img)] = 65000
    npu_img[npu_img < -65000] = -65000
    npu_img[npu_img > 65000] = 65000
    cpu_img[np.isnan(cpu_img)] = 0
    cpu_img[np.isneginf(cpu_img)] = -65000
    cpu_img[np.isinf(cpu_img)] = 65000
    cpu_img[cpu_img < -65000] = -65000
    cpu_img[cpu_img > 65000] = 65000

    diff_img = cpu_img - npu_img

    return float(1 - np.sqrt(np.sum(diff_img * diff_img) / (len(cpu_img.flatten()))) / (max(0.001, max(0, (np.max(cpu_img))) - min(0, np.min(cpu_img)), max(0, (np.max(npu_img))) - min(0, np.min(npu_img)))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("left", help="left file")
    parser.add_argument("right", help="right file")

    args = parser.parse_args()

    diff = compare_blobs(args.left, args.right)
    print(diff)
