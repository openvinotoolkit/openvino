# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from mo.utils.cli_parser import get_all_cli_parser

from ngraph.frontend import FrontEndManager  # pylint: disable=no-name-in-module,import-error


if __name__ == "__main__":
    from mo.main import main
    fem = FrontEndManager()
    sys.exit(main(get_all_cli_parser(fem), fem, 'pdpd'))
