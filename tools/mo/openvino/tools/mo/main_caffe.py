# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.tools.mo.utils.cli_parser import get_caffe_cli_parser

if __name__ == "__main__":
    from openvino.tools.mo.main import main
    sys.exit(main(get_caffe_cli_parser(), 'caffe'))
