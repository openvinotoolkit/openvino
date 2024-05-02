# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.frontend import (  # pylint: disable=no-name-in-module,import-error
    FrontEndManager,
)

from openvino.tools.mo.utils.cli_parser import (  # pylint: disable=no-name-in-module,import-error
    get_all_cli_parser,
)

if __name__ == "__main__":
    from openvino.tools.mo.main import main

    sys.exit(main(get_all_cli_parser(), "paddle"))
