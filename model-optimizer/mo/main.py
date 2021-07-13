# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging as log
import os
import platform
import subprocess
import sys
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np

try:
    import openvino_telemetry as tm
except ImportError:
    import mo.utils.telemetry_stub as tm

from extensions.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement, NormalizeTI
from mo.back.ie_ir_ver_2.emitter import append_ir_info
from mo.moc_frontend.pipeline import moc_pipeline
from mo.moc_frontend.serialize import moc_emit_ir
from mo.graph.graph import Graph
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.pipeline.common import prepare_emit_ir, get_ir_version
from mo.pipeline.unified import unified_pipeline
from mo.utils import import_extensions
from mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_model_name, \
    get_common_cli_options, get_caffe_cli_options, get_tf_cli_options, get_mxnet_cli_options, get_kaldi_cli_options, \
    get_onnx_cli_options, get_mean_scale_dictionary, parse_tuple_pairs, get_freeze_placeholder_values, get_meta_info, \
    parse_transform, check_available_transforms
from mo.utils.error import Error, FrameworkError
from mo.utils.find_ie_version import find_ie_version
from mo.utils.get_ov_update_message import get_ov_update_message
from mo.utils.guess_framework import deduce_framework_by_namespace
from mo.utils.logger import init_logger
from mo.utils.model_analysis import AnalysisResults
from mo.utils.utils import refer_to_faq_msg
from mo.utils.telemetry_utils import send_params_info, send_framework_info
from mo.utils.version import get_version, get_simplified_mo_version, get_simplified_ie_version
from mo.utils.versions_checker import check_requirements  # pylint: disable=no-name-in-module

# pylint: disable=no-name-in-module,import-error
from ngraph.frontend import FrontEndManager


def replace_ext(name: str, old: str, new: str):
    base, ext = os.path.splitext(name)
    log.debug("base: {}, ext: {}".format(base, ext))
    if ext == old:
        return base + new


def print_argv(argv: argparse.Namespace, is_caffe: bool, is_tf: bool, is_mxnet: bool, is_kaldi: bool, is_onnx: bool,
               model_name: str):
    print('Model Optimizer arguments:')
    props = OrderedDict()
    props['common_args'] = get_common_cli_options(model_name)
    if is_caffe:
        props['caffe_args'] = get_caffe_cli_options()
    if is_tf:
        props['tf_args'] = get_tf_cli_options()
    if is_mxnet:
        props['mxnet_args'] = get_mxnet_cli_options()
    if is_kaldi:
        props['kaldi_args'] = get_kaldi_cli_options()
    if is_onnx:
        props['onnx_args'] = get_onnx_cli_options()

    framework_specifics_map = {
        'common_args': 'Common parameters:',
        'caffe_args': 'Caffe specific parameters:',
        'tf_args': 'TensorFlow specific parameters:',
        'mxnet_args': 'MXNet specific parameters:',
        'kaldi_args': 'Kaldi specific parameters:',
        'onnx_args': 'ONNX specific parameters:',
    }

    lines = []
    for key in props:
        lines.append(framework_specifics_map[key])
        for (op, desc) in props[key].items():
            if isinstance(desc, list):
                lines.append('\t{}: \t{}'.format(desc[0], desc[1](getattr(argv, op, 'NONE'))))
            else:
                if op == 'k':
                    default_path = os.path.join(os.path.dirname(sys.argv[0]),
                                                'extensions/front/caffe/CustomLayersMapping.xml')
                    if getattr(argv, op, 'NONE') == default_path:
                        lines.append('\t{}: \t{}'.format(desc, 'Default'))
                        continue
                lines.append('\t{}: \t{}'.format(desc, getattr(argv, op, 'NONE')))
    print('\n'.join(lines), flush=True)


def prepare_ir(argv: argparse.Namespace):
    is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx = deduce_framework_by_namespace(argv)

    fem = argv.feManager
    new_front_ends = []
    if fem is not None:  # in future, check of 'use_legacy_frontend' in argv can be added here
        new_front_ends = fem.get_available_front_ends()

    if not any([is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx]):
        frameworks = ['tf', 'caffe', 'mxnet', 'kaldi', 'onnx']
        frameworks = list(set(frameworks + new_front_ends))
        if argv.framework not in frameworks:
            raise Error('Framework {} is not a valid target. Please use --framework with one from the list: {}. ' +
                        refer_to_faq_msg(15), argv.framework, frameworks)

    if is_tf and not argv.input_model and not argv.saved_model_dir and not argv.input_meta_graph:
        raise Error('Path to input model or saved model dir is required: use --input_model, --saved_model_dir or '
                    '--input_meta_graph')
    elif is_mxnet and not argv.input_model and not argv.input_symbol and not argv.pretrained_model_name:
        raise Error('Path to input model or input symbol or pretrained_model_name is required: use --input_model or '
                    '--input_symbol or --pretrained_model_name')
    elif is_caffe and not argv.input_model and not argv.input_proto:
        raise Error('Path to input model or input proto is required: use --input_model or --input_proto')
    elif (is_kaldi or is_onnx) and not argv.input_model:
        raise Error('Path to input model is required: use --input_model.')

    log.debug(str(argv))
    log.debug("Model Optimizer started")

    model_name = "<UNKNOWN_NAME>"
    if argv.model_name:
        model_name = argv.model_name
    elif argv.input_model:
        model_name = get_model_name(argv.input_model)
    elif is_tf and argv.saved_model_dir:
        model_name = "saved_model"
    elif is_tf and argv.input_meta_graph:
        model_name = get_model_name(argv.input_meta_graph)
    elif is_mxnet and argv.input_symbol:
        model_name = get_model_name(argv.input_symbol)
    argv.model_name = model_name

    log.debug('Output model name would be {}{{.xml, .bin}}'.format(argv.model_name))

    # if --input_proto is not provided, try to retrieve another one
    # by suffix substitution from model file name
    if is_caffe and not argv.input_proto:
        argv.input_proto = replace_ext(argv.input_model, '.caffemodel', '.prototxt')

        if not argv.input_proto:
            raise Error("Cannot find prototxt file: for Caffe please specify --input_proto - a " +
                        "protobuf file that stores topology and --input_model that stores " +
                        "pretrained weights. " +
                        refer_to_faq_msg(20))
        log.info('Deduced name for prototxt: {}'.format(argv.input_proto))

    if not argv.silent:
        print_argv(argv, is_caffe, is_tf, is_mxnet, is_kaldi, is_onnx, argv.model_name)

    # This try-except is additional reinsurance that the IE
    # dependency search does not break the MO pipeline
    def raise_ie_not_found():
        raise Error("Could not find the Inference Engine or nGraph Python API.\n"
                    "Consider building the Inference Engine and nGraph Python APIs from sources or try to install OpenVINO (TM) Toolkit using \"install_prerequisites.{}\"".format(
                    "bat" if sys.platform == "windows" else "sh"))
    try:
        if not find_ie_version(silent=argv.silent):
            raise_ie_not_found()
    except Exception as e:
        raise_ie_not_found()

    # This is just to check that transform key is valid and transformations are available
    check_available_transforms(parse_transform(argv.transform))

    if argv.legacy_ir_generation and len(argv.transform) != 0:
        raise Error("--legacy_ir_generation and --transform keys can not be used at the same time.")

    use_legacy_fe = argv.framework not in new_front_ends
    # For C++ frontends there is no specific python installation requirements, thus check only generic ones
    ret_code = check_requirements(framework=argv.framework if use_legacy_fe else None)
    if ret_code:
        raise Error('check_requirements exit with return code {}'.format(ret_code))

    if is_tf and argv.tensorflow_use_custom_operations_config is not None:
        argv.transformations_config = argv.tensorflow_use_custom_operations_config

    if is_caffe and argv.mean_file and argv.mean_values:
        raise Error('Both --mean_file and mean_values are specified. Specify either mean file or mean values. ' +
                    refer_to_faq_msg(17))
    elif is_caffe and argv.mean_file and argv.mean_file_offsets:
        values = get_tuple_values(argv.mean_file_offsets, t=int, num_exp_values=2)
        mean_file_offsets = np.array([int(x) for x in values[0].split(',')])
        if not all([offset >= 0 for offset in mean_file_offsets]):
            raise Error("Negative value specified for --mean_file_offsets option. "
                        "Please specify positive integer values in format '(x,y)'. " +
                        refer_to_faq_msg(18))
        argv.mean_file_offsets = mean_file_offsets

    if argv.scale and argv.scale_values:
        raise Error(
            'Both --scale and --scale_values are defined. Specify either scale factor or scale values per input ' +
            'channels. ' + refer_to_faq_msg(19))

    if argv.scale and argv.scale < 1.0:
        log.error("The scale value is less than 1.0. This is most probably an issue because the scale value specifies "
                  "floating point value which all input values will be *divided*.", extra={'is_warning': True})

    if argv.input_model and (is_tf and argv.saved_model_dir):
        raise Error('Both --input_model and --saved_model_dir are defined. '
                    'Specify either input model or saved model directory.')
    if is_tf:
        if argv.saved_model_tags is not None:
            if ' ' in argv.saved_model_tags:
                raise Error('Incorrect saved model tag was provided. Specify --saved_model_tags with no spaces in it')
            argv.saved_model_tags = argv.saved_model_tags.split(',')

    argv.output = argv.output.split(',') if argv.output else None

    argv.placeholder_shapes, argv.placeholder_data_types = get_placeholder_shapes(argv.input, argv.input_shape,
                                                                                  argv.batch)

    mean_values = parse_tuple_pairs(argv.mean_values)
    scale_values = parse_tuple_pairs(argv.scale_values)
    mean_scale = get_mean_scale_dictionary(mean_values, scale_values, argv.input)
    argv.mean_scale_values = mean_scale

    if not os.path.exists(argv.output_dir):
        try:
            os.makedirs(argv.output_dir)
        except PermissionError as e:
            raise Error("Failed to create directory {}. Permission denied! " +
                        refer_to_faq_msg(22),
                        argv.output_dir) from e
    else:
        if not os.access(argv.output_dir, os.W_OK):
            raise Error("Output directory {} is not writable for current user. " +
                        refer_to_faq_msg(22), argv.output_dir)

    log.debug("Placeholder shapes : {}".format(argv.placeholder_shapes))

    if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
        extensions = argv.extensions.split(',')
    else:
        extensions = None

    argv.freeze_placeholder_with_value, argv.input = get_freeze_placeholder_values(argv.input,
                                                                                   argv.freeze_placeholder_with_value)
    if is_tf:
        from mo.front.tf.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_caffe:
        send_framework_info('caffe')
        from mo.front.caffe.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_mxnet:
        send_framework_info('mxnet')
        from mo.front.mxnet.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_kaldi:
        send_framework_info('kaldi')
        from mo.front.kaldi.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_onnx:  # in future check of 'use_legacy_frontend' can be added here
        send_framework_info('onnx')
        from mo.front.onnx.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)

    graph = None
    ngraph_function = None

    # In future check of use_legacy_frontend option can be added here
    if argv.feManager is None or argv.framework not in new_front_ends:
        graph = unified_pipeline(argv)
    else:
        ngraph_function = moc_pipeline(argv)
    return graph, ngraph_function


def emit_ir(graph: Graph, argv: argparse.Namespace):
    NormalizeTI().find_and_replace_pattern(graph)
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    if 'feManager' in argv:
        del argv.feManager

    mean_data = deepcopy(graph.graph['mf']) if 'mf' in graph.graph else None
    input_names = deepcopy(graph.graph['input_names']) if 'input_names' in graph.graph else []

    prepare_emit_ir(graph=graph,
                    data_type=graph.graph['cmd_params'].data_type,
                    output_dir=argv.output_dir,
                    output_model_name=argv.model_name,
                    mean_data=mean_data,
                    input_names=input_names,
                    meta_info=get_meta_info(argv),
                    use_temporary_path=True)

    # This graph cleanup is required to avoid double memory consumption
    graph.clear()

    if not (argv.framework == 'tf' and argv.tensorflow_custom_operations_config_update):
        output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
        orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))

        return_code = "not executed"
        # This try-except is additional reinsurance that the IE
        # dependency search does not break the MO pipeline
        try:
            if not argv.legacy_ir_generation:
                path_to_offline_transformations = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'back',
                                                               'offline_transformations.py')
                status = subprocess.run([sys.executable, path_to_offline_transformations,
                                         "--input_model", orig_model_name,
                                         "--framework", argv.framework,
                                         "--transform", argv.transform], env=os.environ)
                return_code = status.returncode
        except Exception as e:
            return_code = "failed"
            log.error(e)

        message = str(dict({
            "platform": platform.system(),
            "mo_version": get_simplified_mo_version(),
            "ie_version": get_simplified_ie_version(env=os.environ),
            "python_version": sys.version,
            "return_code": return_code
        }))
        t = tm.Telemetry()
        t.send_event('mo', 'offline_transformations_status', message)

        if return_code != 0:
            raise Error("offline transformations step has failed.")

        for suf in [".xml", ".bin", ".mapping"]:
            # remove existing files
            path_to_file = orig_model_name + "_tmp" + suf
            if os.path.exists(path_to_file):
                os.remove(path_to_file)

        # add meta information to IR
        append_ir_info(file=orig_model_name,
                       meta_info=get_meta_info(argv),
                       mean_data=mean_data,
                       input_names=input_names)

        print('[ SUCCESS ] Generated IR version {} model.'.format(get_ir_version(argv)))
        print('[ SUCCESS ] XML file: {}.xml'.format(orig_model_name))
        print('[ SUCCESS ] BIN file: {}.bin'.format(orig_model_name))

    return 0


def driver(argv: argparse.Namespace):
    init_logger(argv.log_level.upper(), argv.silent)

    start_time = datetime.datetime.now()

    graph, ngraph_function = prepare_ir(argv)
    if graph is not None:
        ret_res = emit_ir(graph, argv)
    else:
        ret_res = moc_emit_ir(ngraph_function, argv)

    if ret_res != 0:
        return ret_res

    elapsed_time = datetime.datetime.now() - start_time
    print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

    try:
        import resource
        mem_usage = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if sys.platform == 'darwin':
            mem_usage = round(mem_usage / 1024)
        print('[ SUCCESS ] Memory consumed: {} MB. '.format(mem_usage))
    except ImportError:
        pass

    return ret_res


def main(cli_parser: argparse.ArgumentParser, fem: FrontEndManager, framework: str):
    telemetry = tm.Telemetry(app_name='Model Optimizer', app_version=get_simplified_mo_version())
    telemetry.start_session('mo')
    telemetry.send_event('mo', 'version', get_simplified_mo_version())
    try:
        # Initialize logger with 'ERROR' as default level to be able to form nice messages
        # before arg parser deliver log_level requested by user
        init_logger('ERROR', False)

        argv = cli_parser.parse_args()
        send_params_info(argv, cli_parser)

        if framework:
            argv.framework = framework
        argv.feManager = fem

        ov_update_message = None
        if not hasattr(argv, 'silent') or not argv.silent:
            ov_update_message = get_ov_update_message()
        ret_code = driver(argv)
        if ov_update_message:
            print(ov_update_message)
        telemetry.send_event('mo', 'conversion_result', 'success')
        telemetry.end_session('mo')
        telemetry.force_shutdown(1.0)
        return ret_code
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

    telemetry.send_event('mo', 'conversion_result', 'fail')
    telemetry.end_session('mo')
    telemetry.force_shutdown(1.0)
    return 1


if __name__ == "__main__":
    from mo.utils.cli_parser import get_all_cli_parser
    fem = FrontEndManager()
    sys.exit(main(get_all_cli_parser(fem), fem, None))
