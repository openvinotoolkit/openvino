#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

if __name__ == "__main__":
    try:
        from openvino.inference_engine import IECore # pylint: disable=import-error
        from openvino.offline_transformations import ApplyMOCTransformations, CheckAPI # pylint: disable=import-error
    except Exception as e:
        print("[ WARNING ] {}".format(e))
        exit(1)

    CheckAPI()