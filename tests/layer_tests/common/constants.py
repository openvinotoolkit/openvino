# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


if 'MO_ROOT' in os.environ:
    mo_bin = os.environ['MO_ROOT']
    if not os.path.exists(mo_bin):
        raise EnvironmentError(
            "Environment variable MO_ROOT points to non existing path {}".format(mo_bin))
else:
    raise EnvironmentError("MO_ROOT variable is not set")

if os.environ.get('OUTPUT_DIR') is not None:
    out_path = os.environ['OUTPUT_DIR']
else:
    script_path = os.path.dirname(os.path.realpath(__file__))
    out_path = os.path.join(script_path, 'out')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

# supported_devices : CPU, GPU, MYRIAD, FPGA
test_device = os.environ.get('TEST_DEVICE', 'CPU;GPU').split(';')
test_precision = os.environ.get('TEST_PRECISION', 'FP32;FP16').split(';')
