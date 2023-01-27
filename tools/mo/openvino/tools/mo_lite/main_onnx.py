# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.tools.mo_lite.utils.cli_parser import get_onnx_cli_parser

if __name__ == "__main__":
    from openvino.tools.mo_lite.main import main

    sys.exit(main(get_onnx_cli_parser(), 'onnx'))
