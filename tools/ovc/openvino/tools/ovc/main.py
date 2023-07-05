# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm
from openvino.tools.ovc.convert_impl import _convert
from openvino.tools.ovc.version import VersionChecker

# pylint: disable=no-name-in-module,import-error
from openvino.runtime import serialize


def main():
    from openvino.tools.ovc.cli_parser import get_all_cli_parser
    ngraph_function, argv = _convert(get_all_cli_parser(), {}, False)
    if ngraph_function is None:
        return 1

    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
    model_path_no_ext = os.path.normpath(os.path.join(output_dir, argv.model_name))
    model_path = model_path_no_ext + '.xml'

    serialize(ngraph_function, model_path.encode('utf-8'), model_path.replace('.xml', '.bin').encode('utf-8'))

    print('[ SUCCESS ] Generated IR version {} model.'.format(VersionChecker().get_ie_simplified_version()))
    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    sys.exit(main())
