#!/usr/bin/python3

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from openvino.tools.benchmark.main import main


if __name__ == "__main__":
    sys.exit(main() or 0)
