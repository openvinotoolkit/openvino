# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from mo.pipeline.common import get_ir_version
from mo.back.ie_ir_ver_2.emitter import append_ir_info
from mo.utils.cli_parser import get_meta_info

from ngraph import Function         # pylint: disable=no-name-in-module,import-error
from ngraph import function_to_cnn  # pylint: disable=no-name-in-module,import-error


def moc_emit_ir(ngraph_function: Function, argv: argparse.Namespace):
    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()

    network = function_to_cnn(ngraph_function)

    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))
    network.serialize(orig_model_name + ".xml", orig_model_name + ".bin")

    del argv.feManager

    # add meta information to IR
    append_ir_info(file=orig_model_name,
                   meta_info=get_meta_info(argv),
                   mean_data=None,
                   input_names=None)

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}.xml'.format(orig_model_name))
    print('[ SUCCESS ] BIN file: {}.bin'.format(orig_model_name))
    return 0
