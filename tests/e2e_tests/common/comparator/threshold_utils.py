# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import logging as log

# default thresholds for comparators
DEFAULT_THRESHOLDS = {
    "FP32": (1e-4, 1e-4, None),
    "BF16": (2, 2, None),
    "FP16": (0.01, 2, None)
}

DEFAULT_IOU_THRESHOLDS = {
    "FP32": 0.9,
    "BF16": 0.8,
    "FP16": 0.8
}

DEFAULT_SSIM_THRESHOLDS = {
    "FP32": 0.99,
    "BF16": 0.9,
    "FP16": 0.9
}

# fallback thresholds if precision not found
FALLBACK_EPS = (1e-4, 1e-4, None)


def get_default_thresholds(precision, device):
    """Get default comparison thresholds (a_eps, r_eps) for specific precision.

    :param precision:   network's precision (e.g. FP16)
    :return:    pair of thresholds (absolute eps, relative eps)
    """
    # setup logger
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    if precision not in DEFAULT_THRESHOLDS:
        log.warning("Specified precision {precision} for comparison thresholds "
                    "not found. Using {fallback} instead.".format(precision=precision,
                                                                  fallback=FALLBACK_EPS))

    #for FPGA FP16 thresholds are used always
    if "FPGA" in device or "HDDL" in device:
        return DEFAULT_THRESHOLDS.get("FP16", FALLBACK_EPS)

    return DEFAULT_THRESHOLDS.get(precision, FALLBACK_EPS)


def get_default_iou_threshold(precision, device):
    """Get default comparison thresholds (a_eps, r_eps) for specific precision.

    :param precision:   network's precision (e.g. FP16)
    :return:    pair of thresholds (absolute eps, relative eps)
    """
    # setup logger
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    if precision not in DEFAULT_IOU_THRESHOLDS:
        log.warning("Specified precision {precision} for comparison thresholds "
                    "not found. Using {fallback} instead.".format(precision=precision,
                                                                  fallback=0.9))

    # for FPGA FP16 thresholds are used always
    if "FPGA" in device or "HDDL" in device:
        return DEFAULT_IOU_THRESHOLDS.get("FP16", FALLBACK_EPS)

    return DEFAULT_IOU_THRESHOLDS.get(precision, 0.9)


def get_default_ssim_threshold(precision, device):
    """Get default comparison thresholds (a_eps, r_eps) for specific precision.

    :param precision:   network's precision (e.g. FP16)
    :return:    pair of thresholds (absolute eps, relative eps)
    """
    # setup logger
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    if precision not in DEFAULT_SSIM_THRESHOLDS:
        log.warning("Specified precision {precision} for comparison thresholds "
                    "not found. Using {fallback} instead.".format(precision=precision,
                                                                  fallback=0.9))
    # for FPGA FP16 thresholds are used always
    if "FPGA" in device or "HDDL" in device:
        return DEFAULT_SSIM_THRESHOLDS.get("FP16", FALLBACK_EPS)

    return DEFAULT_SSIM_THRESHOLDS.get(precision, 0.9)
