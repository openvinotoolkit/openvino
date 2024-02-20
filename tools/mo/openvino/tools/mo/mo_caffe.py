#!/usr/bin/env python3

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


if __name__ == "__main__":
    from openvino.tools.mo.subprocess_main import subprocess_main
    from openvino.tools.mo.utils.telemetry_utils import init_mo_telemetry
    init_mo_telemetry()
    subprocess_main(framework='caffe')
