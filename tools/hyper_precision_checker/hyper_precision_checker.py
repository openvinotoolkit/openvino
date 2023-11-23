#!/usr/bin/python3

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from openvino.tools.hyper_precision_checker.main import main


if __name__ == "__main__":
    sys.exit(main() or 0)
