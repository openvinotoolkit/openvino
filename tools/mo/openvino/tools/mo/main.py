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
from convert import convert
from serialize import serialize


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


def main(cli_parser: argparse.ArgumentParser, fem: FrontEndManager, framework: str):
    argv = cli_parser.parse_args()
    model_name = get_model_name_from_args(argv)
    ngraph_function = convert(
        argv.input_model,
        framework,
        model_name,
        argv.input_shape,
        argv.scale,
        argv.reverse_input_channels,
        argv.log_level,
        argv.input,
        argv.output,
        argv.mean_values,
        argv.scale_values,
        argv.source_layout,
        argv.target_layout,
        argv.layout,
        argv.transform,
        argv.extensions,
        argv.batch,
        argv.silent,
        argv.static_shape,
        argv.progress,
        argv.stream_output,
        argv.transformations_config,
        argv.use_new_frontend,
        argv.use_legacy_frontend,
        argv.disable_omitting_optional,
        argv.enable_flattening_nested_params,
        argv.input_model_is_text,
        argv.input_checkpoint,
        argv.input_meta_graph,
        argv.saved_model_dir,
        argv.saved_model_tags,
        argv.tensorflow_custom_operations_config_update,
        argv.tensorflow_object_detection_api_pipeline_config,
        argv.tensorboard_logdir,
        argv.tensorflow_custom_layer_libraries,
        argv.input_proto,
        argv.caffe_parser_path,
        argv.k,
        argv.input_symbol,
        argv.nd_prefix_name,
        argv.pretrained_model_name,
        argv.save_params_from_nd,
        argv.legacy_mxnet_model,
        argv.enable_ssd_gluoncv,
        argv.counts,
        argv.remove_output_softmax,
        argv.remove_memory)

    if ngraph_function is None:
        return 1

    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
    model_path = os.path.normpath(os.path.join(output_dir, model_name + '.xml'))

    serialize(ngraph_function, model_path, argv)

    print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
    print('[ SUCCESS ] XML file: {}'.format(model_path))
    print('[ SUCCESS ] BIN file: {}'.format(model_path.replace('.xml', '.bin')))
    return 0


if __name__ == "__main__":
    from openvino.tools.mo.utils.cli_parser import get_all_cli_parser
    fe_manager = FrontEndManager()
    sys.exit(main(get_all_cli_parser(fe_manager), fe_manager, None))
