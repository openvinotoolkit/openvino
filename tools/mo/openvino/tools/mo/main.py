# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.tools.mo.utils.cli_parser import get_model_name

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager
from openvino.runtime import serialize
from convert import convert


def get_model_name_from_args(argv: argparse.Namespace):
    model_name = "<UNKNOWN_NAME>"
    if argv.model_name:
        model_name = argv.model_name
    elif argv.input_model:
        model_name = get_model_name(argv.input_model)
    elif argv.saved_model_dir:
        model_name = "saved_model"
    elif argv.input_meta_graph:
        model_name = get_model_name(argv.input_meta_graph)
    elif argv.input_symbol:
        model_name = get_model_name(argv.input_symbol)
    argv.model_name = model_name
    return model_name


def main(cli_parser: argparse.ArgumentParser):
    argv = cli_parser.parse_args()
    argv.model_name = get_model_name_from_args(argv)
    argv = vars(argv)
    ngraph_function = convert(**argv)

    if ngraph_function is None:
        return 1

    output_dir = argv['output_dir'] if argv['output_dir'] != '.' else os.getcwd()
    model_path = os.path.normpath(os.path.join(output_dir, argv['model_name'] + '.xml'))

    # Ticket for fixing: 88606
    print('[ WARNING ] MO Meta data is not serialized to IR.'.format(get_ir_version(argv)))

    serialize(ngraph_function, model_path.encode('utf-8'), model_path.replace('.xml', '.bin').encode('utf-8'))

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    from openvino.tools.mo.utils.cli_parser import get_all_cli_parser
    sys.exit(main(get_all_cli_parser(FrontEndManager())))
