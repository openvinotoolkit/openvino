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
import sys
import logging as log

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

# default thresholds for comparators
DEFAULT_THRESHOLDS = {
    "FP32": (1e-4, 1e-4),
    "FP16": (3, 3)
}

DEFAULT_IOU_THRESHOLDS = {
    "FP32": 0.9,
    "FP16": 0.8
}


def get_default_thresholds(params):
    """Get default comparison thresholds (a_eps, r_eps) for specific precision.

    :param params:   all params to sample
    :return:         pair of thresholds (absolute eps, relative eps)
    """
    # Check if some model of all is FP16. If so - use thresholds for FP16
    models_list = [v.lower() for k,v in params.items() if k == 'm' or 'm_' in k]
    precision = 'FP16' if any([l for l in models_list if 'fp16' in l]) else 'FP32'
    return DEFAULT_THRESHOLDS.get(precision)

def get_default_iou_threshold(params):
    """Get default comparison thresholds (a_eps, r_eps) for specific precision.

    :param params:   all params to sample
    :return:         absolute eps threshold
    """
    # Check if some model of all is FP16. If so - use thresholds for FP16
    models_list = [v.lower() for k,v in params.items() if k == 'm' or 'm_' in k]
    precision = 'FP16' if any([l for l in models_list if 'fp16' in l]) else 'FP32'
    return DEFAULT_IOU_THRESHOLDS.get(precision)