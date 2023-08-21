# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm
from openvino.tools.mo.convert_impl import _convert
from openvino.tools.mo.pipeline.common import get_ir_version

# pylint: disable=no-name-in-module,import-error
from openvino.runtime import serialize


def main(cli_parser: argparse.ArgumentParser, framework=None):
    ngraph_function, argv = _convert(cli_parser, framework, {}, False)
    if ngraph_function is None:
        return 1

    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
    model_path_no_ext = os.path.normpath(os.path.join(output_dir, argv.model_name))
    model_path = model_path_no_ext + '.xml'

    serialize(ngraph_function, model_path.encode('utf-8'), model_path.replace('.xml', '.bin').encode('utf-8'))

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    from openvino.tools.mo.utils.cli_parser import get_all_cli_parser  # pylint: disable=no-name-in-module,import-error
    sys.exit(main(get_all_cli_parser(), None))
