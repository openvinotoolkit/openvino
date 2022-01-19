#!/usr/bin/python3

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from openvino.tools.cross_check_tool.cross_check_tool import main


if __name__ == "__main__":
    sys.exit(main() or 0)
