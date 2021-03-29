#!/usr/bin/env python3

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from mo.utils.versions_checker import check_python_version

if __name__ == "__main__":
    ret_code = check_python_version()
    if ret_code:
        sys.exit(ret_code)

    from mo.main import main
    from mo.utils.cli_parser import get_mxnet_cli_parser

    sys.exit(main(get_mxnet_cli_parser(), 'mxnet'))
