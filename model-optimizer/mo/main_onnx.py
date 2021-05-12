# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from mo.utils.cli_parser import get_onnx_cli_parser

if __name__ == "__main__":
    from mo.main import main
    sys.exit(main(get_onnx_cli_parser(), 'onnx'))
