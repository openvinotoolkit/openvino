# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.tools.mo.utils.cli_parser import get_tf_cli_parser  # pylint: disable=no-name-in-module,import-error

if __name__ == "__main__":
    from openvino.tools.mo.main import main
    sys.exit(main(get_tf_cli_parser(), 'tf'))
