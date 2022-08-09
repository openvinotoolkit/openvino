# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import logging as log

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

from openvino.tools.mo.convert import convert
from openvino.tools.mo.pipeline.common import get_ir_version
from openvino.tools.mo.utils.cli_parser import get_model_name, get_meta_info
from openvino.tools.mo.utils.logger import init_logger
from openvino.tools.mo.utils.error import Error, FrameworkError
import traceback
from openvino.tools.mo.utils.model_analysis import AnalysisResults
from openvino.tools.mo.back.ie_ir_ver_2.emitter import append_ir_info

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager
from openvino.runtime import serialize


def get_model_name_from_args(argv: argparse.Namespace):
    model_name = "<UNKNOWN_NAME>"
    if hasattr(argv, 'model_name'):
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

    # Initialize logger with 'ERROR' as default level to be able to form nice messages
    # before arg parser deliver log_level requested by user
    init_logger('ERROR', False)

    ngraph_function = None
    try:
        ngraph_function = convert(**argv)
    except (FileNotFoundError, NotADirectoryError) as e:
        log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
        log.debug(traceback.format_exc())
    except Error as err:
        analysis_results = AnalysisResults()
        if analysis_results.get_messages() is not None:
            for el in analysis_results.get_messages():
                log.error(el, extra={'analysis_info': True})
        log.error(err)
        log.debug(traceback.format_exc())
    except FrameworkError as err:
        log.error(err, extra={'framework_error': True})
        log.debug(traceback.format_exc())
    except Exception as err:
        log.error("-------------------------------------------------")
        log.error("----------------- INTERNAL ERROR ----------------")
        log.error("Unexpected exception happened.")
        log.error("Please contact Model Optimizer developers and forward the following information:")
        log.error(str(err))
        log.error(traceback.format_exc())
        log.error("---------------- END OF BUG REPORT --------------")
        log.error("-------------------------------------------------")

    if ngraph_function is None:
        return 1

    output_dir = argv['output_dir'] if argv['output_dir'] != '.' else os.getcwd()
    model_path = os.path.normpath(os.path.join(output_dir, argv['model_name'] + '.xml'))

    serialize(ngraph_function, model_path.encode('utf-8'), model_path.replace('.xml', '.bin').encode('utf-8'))

    # add meta information to IR
    append_ir_info(file=model_path, meta_info=get_meta_info(argv))

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    from openvino.tools.mo.utils.cli_parser import get_all_cli_parser
    sys.exit(main(get_all_cli_parser(FrontEndManager())))
