# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from mo.utils.cli_parser import get_all_cli_parser

if __name__ == "__main__":
    from mo.main import main
    from mo.front_ng.frontendmanager_wrapper import create_fem
    fem = create_fem()
    sys.exit(main(get_all_cli_parser(fem), fem, 'pdpd'))
