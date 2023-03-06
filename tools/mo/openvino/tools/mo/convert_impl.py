# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging as log
import os
import platform
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

from openvino.tools.mo.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement, NormalizeTI
from openvino.tools.mo.moc_frontend.check_config import legacy_transformations_config_used, \
    tensorflow_custom_operations_config_update_used, new_extensions_used
from openvino.tools.mo.moc_frontend.pipeline import moc_pipeline
from openvino.tools.mo.moc_frontend.serialize import moc_emit_ir
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.pipeline.common import prepare_emit_ir
from openvino.tools.mo.pipeline.unified import unified_pipeline
from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.utils.cli_parser import check_available_transforms, \
    get_advanced_cli_options, get_available_front_ends, get_caffe_cli_options, \
    get_common_cli_options, get_freeze_placeholder_values, get_kaldi_cli_options, get_layout_values, \
    get_mean_scale_dictionary, get_mxnet_cli_options, get_onnx_cli_options, \
    get_placeholder_shapes, get_tf_cli_options, parse_transform, parse_tuple_pairs, \
    mo_convert_params, get_model_name_from_args, depersonalize

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.find_ie_version import find_ie_version
from openvino.tools.mo.utils.guess_framework import deduce_legacy_frontend_by_namespace
from openvino.tools.mo.utils.logger import init_logger, progress_printer
from openvino.tools.mo.utils.utils import refer_to_faq_msg
from openvino.tools.mo.utils.telemetry_utils import send_params_info, send_framework_info
from openvino.tools.mo.utils.version import get_simplified_mo_version, get_simplified_ie_version, get_version
from openvino.tools.mo.utils.versions_checker import check_requirements  # pylint: disable=no-name-in-module
from openvino.tools.mo.utils.telemetry_utils import get_tid
from openvino.tools.mo.moc_frontend.check_config import legacy_extensions_used
from openvino.tools.mo.moc_frontend.pytorch_frontend_utils import get_pytorch_decoder, convert_pytorch_via_onnx
from openvino.tools.mo.moc_frontend.shape_utils import parse_input_shapes, get_static_shape

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager, OpConversionFailure, ProgressReporterExtension, TelemetryExtension
from openvino.runtime import get_version as get_rt_version


def load_extensions(argv: argparse.Namespace, is_tf: bool, is_caffe: bool, is_mxnet: bool, is_kaldi: bool,
                    is_onnx: bool):
    extensions = None
    if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
        extensions = argv.extensions
    if is_tf:
        from openvino.tools.mo.front.tf.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_caffe:
        send_framework_info('caffe')
        from openvino.tools.mo.front.caffe.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_mxnet:
        send_framework_info('mxnet')
        from openvino.tools.mo.front.mxnet.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_kaldi:
        send_framework_info('kaldi')
        from openvino.tools.mo.front.kaldi.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
    elif is_onnx:
        send_framework_info('onnx')
        from openvino.tools.mo.front.onnx.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)


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
    props['advanced_args'] = get_advanced_cli_options()
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
        'advanced_args': 'Advanced parameters:',
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
                                                'openvino/tools/mo/front/caffe/CustomLayersMapping.xml')
                    if getattr(argv, op, 'NONE') == default_path:
                        lines.append('\t{}: \t{}'.format(desc, 'Default'))
                        continue
                lines.append('\t{}: \t{}'.format(desc, getattr(argv, op, 'NONE')))
    print('\n'.join(lines), flush=True)


def arguments_post_parsing(argv: argparse.Namespace):
    use_legacy_frontend = argv.use_legacy_frontend
    use_new_frontend = argv.use_new_frontend

    if use_new_frontend and use_legacy_frontend:
        raise Error('Options --use_new_frontend and --use_legacy_frontend must not be used simultaneously '
                    'in the Model Optimizer command-line')

    moc_front_end, available_moc_front_ends = get_moc_frontends(argv)

    if not moc_front_end and use_new_frontend:
        raise Error('Option --use_new_frontend is specified but the Model Optimizer is unable to find new frontend. '
                    'Please ensure that your environment contains new frontend for the input model format or '
                    'try to convert the model without specifying --use_new_frontend option.')

    is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx = \
        deduce_legacy_frontend_by_namespace(argv) if not moc_front_end else [False, False, False, False, False]

    is_legacy_frontend = any([is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx])
    if not is_legacy_frontend and use_legacy_frontend:
        raise Error('Option --use_legacy_frontend is specified but Model Optimizer does not have legacy frontend '
                    'for the input model format. Please try to convert the model without specifying --use_legacy_frontend option.')

    # handle a default case, i.e. use_new_frontend and use_legacy_frontend are not specified, when no frontend is found
    if not is_legacy_frontend and not moc_front_end:
        legacy_frameworks = ['tf', 'caffe', 'mxnet', 'kaldi', 'onnx']
        frameworks = list(set(legacy_frameworks + available_moc_front_ends))
        if not argv.framework:
            raise Error('Framework name can not be deduced from the given options: {}={}. '
                        'Please use --framework with one from the list: {}.',
                        '--input_model', argv.input_model, frameworks)
        elif argv.framework not in frameworks:
            if argv.framework == 'ir':
                raise Error('OpenVINO IR is passed as input_model in convert_model/mo, the IR doesn\'t need '
                            'conversion, please use it in runtime for inference with read_model/compile_model.')
            raise Error('Framework {} is not a valid target. Please use --framework with one from the list: {}. ' +
                        refer_to_faq_msg(15), argv.framework, frameworks)

    if is_legacy_frontend:
        if new_extensions_used(argv):
            raise Error('New kind of extensions used on legacy path')

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

    log.debug("Model Optimizer started")

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
                    "Consider building the Inference Engine and nGraph Python APIs from sources or "
                    "try to install OpenVINO (TM) Toolkit using pip \npip install openvino")

    try:
        if not find_ie_version(silent=argv.silent):
            raise_ie_not_found()
    except Exception as e:
        log.error(e)
        raise_ie_not_found()

    # Turn off compression only if it's disabled explicitly by --compress_to_fp16=False or --data_type=FP32.
    # By default, in all other cases compression is enabled
    if ('data_type' in argv and argv.data_type in ['FP32', 'float']) or \
            ('compress_to_fp16' in argv and argv.compress_to_fp16 is False):
        argv.compress_fp16 = False
    else:
        argv.compress_fp16 = True
    argv.data_type = 'FP32'  # if compression was enabled will be restored back to 'FP16' after apply_offline_transformations

    # This is just to check that transform key is valid and transformations are available
    check_available_transforms(parse_transform(argv.transform))

    # For C++ frontends there are no specific Python installation requirements, check only generic ones
    if moc_front_end:
        ret_code = check_requirements(silent=argv.silent)
    else:
        ret_code = check_requirements(framework=argv.framework, silent=argv.silent)
    if ret_code:
        raise Error('check_requirements exited with return code {}'.format(ret_code))

    if hasattr(argv, 'tensorflow_use_custom_operations_config') and \
            argv.tensorflow_use_custom_operations_config is not None:
        # update command-line arguments even for new TensorFlow Frontend
        # because it should fallback to the Legacy Frontend in this case
        argv.transformations_config = argv.tensorflow_use_custom_operations_config

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

    inputs_list, argv.placeholder_shapes, argv.placeholder_data_types = get_placeholder_shapes(
        argv.input, argv.input_shape, argv.batch)
    argv.inputs_list = inputs_list

    mean_values = parse_tuple_pairs(argv.mean_values)
    scale_values = parse_tuple_pairs(argv.scale_values)
    mean_scale = get_mean_scale_dictionary(mean_values, scale_values, argv.input)
    argv.mean_scale_values = mean_scale
    argv.layout_values = get_layout_values(argv.layout, argv.source_layout, argv.target_layout)

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

    argv.freeze_placeholder_with_value, argv.input = get_freeze_placeholder_values(argv.input,
                                                                                   argv.freeze_placeholder_with_value)

    load_extensions(argv, is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx)

    return argv


def check_fallback(argv: argparse.Namespace):
    fallback_reasons = {}

    # Some frontend such as PDPD does not have legacy path so it has no reasons to fallback
    if not any(deduce_legacy_frontend_by_namespace(argv)):
        return fallback_reasons

    if argv.use_new_frontend:
        return fallback_reasons

    fallback_reasons['extensions'] = legacy_extensions_used
    fallback_reasons['transformations_config'] = legacy_transformations_config_used
    fallback_reasons['tensorflow_custom_operations_config_update'] = tensorflow_custom_operations_config_update_used

    reasons = [reason for reason, is_applicable in fallback_reasons.items() if is_applicable(argv)]
    return reasons


def update_fallback_with_conversion_error(use_new_frontend: bool, is_tf: bool, ex_msg: str, fallback_reasons: list):
    import re
    if not is_tf:
        # this sort of fallback is only used by TensorFlow Frontend
        return False

    if use_new_frontend:
        # this option forces to use new TensorFlow Frontend
        # so it is not possible for the fallback
        return False

    # for TensorFlow FE we have a set of operations that should lead to the fallback to the legacy
    conversion_error_re = r"^(\[TensorFlow\ Frontend\]\ Internal\ error\:\ No\ translator\ found\ for\ )(\w+)(\ node\.)$"
    conversion_error_match = re.findall(conversion_error_re, ex_msg, re.MULTILINE)
    fallback_operations = [
        # corresponds to TF1 While operation
        "TensorArrayScatterV3", "TensorArrayV3", "TensorArraySizeV3", "TensorArrayGatherV3",
        "LoopCond", "Enter", "NextIteration", "Exit",
        # corresponds to TF1 If and TF1 While operations
        "Switch", "Merge",
        # corresponds to operations with complex tensors
        "FFT", "FFT2D", "FFT3D", "IFFT", "IFFT2D", "IFFT3D",
        "RFFT", "RFFT2D", "RFFT3D", "IRFFT", "IRFFT2D", "IRFFT3D",
        "Complex", "ComplexAbs", "Real", "Imag",
    ]
    if len(conversion_error_match) < 1 or len(conversion_error_match[0]) != 3 or \
            conversion_error_match[0][1] not in fallback_operations:
        return False

    fallback_reasons.append("Unsupported operation: " + conversion_error_match[0][1])
    return True


def get_default_frontends():
    # Set which frontend to use by default, values should be 'new' or 'legacy'
    default_frontends = {
        'onnx': 'new',
        'tf': 'new'
    }
    return default_frontends


def get_moc_frontends(argv: argparse.Namespace):
    fem = argv.feManager

    # Read user flags:
    use_legacy_frontend = argv.use_legacy_frontend
    use_new_frontend = argv.use_new_frontend

    if not fem or use_legacy_frontend:
        return None, []

    available_moc_front_ends = get_available_front_ends(fem)

    if not argv.framework and argv.input_model:
        moc_front_end = fem.load_by_model(argv.input_model)
        if not moc_front_end:
            return None, available_moc_front_ends
        argv.framework = moc_front_end.get_name()
    elif argv.framework in available_moc_front_ends:
        moc_front_end = fem.load_by_framework(argv.framework)
    else:
        return None, []

    default_frontends = get_default_frontends()
    # Disable MOC frontend if default is set to legacy and no user override
    if default_frontends.get(moc_front_end.get_name()) == 'legacy' and not use_new_frontend:
        return None, available_moc_front_ends

    # This check as a workaround to skip IR frontend
    if not moc_front_end.get_name() in available_moc_front_ends:
        return None, available_moc_front_ends

    return moc_front_end, available_moc_front_ends


def prepare_ir(argv: argparse.Namespace):
    # TODO: remove this workaround once new TensorFlow frontend supports non-frozen formats: checkpoint, MetaGraph, and SavedModel
    # Now it converts all TensorFlow formats to the frozen .pb format in case new TensorFlow frontend
    is_tf, _, _, _, _ = deduce_legacy_frontend_by_namespace(argv)
    argv = arguments_post_parsing(argv)
    t = tm.Telemetry()
    graph = None
    ngraph_function = None
    fallback_reasons = []
    moc_front_end, available_moc_front_ends = get_moc_frontends(argv)
    if moc_front_end:
        fallback_reasons = check_fallback(argv)
        if len(fallback_reasons) == 0:
            path_to_aux_pb = None
            orig_argv_values = {"input_model": argv.input_model, "model_name": argv.model_name}
            if not argv.use_legacy_frontend and is_tf:
                from openvino.tools.mo.front.tf.loader import convert_to_pb
                path_to_aux_pb = convert_to_pb(argv)
            try:
                t.send_event("mo", "conversion_method", moc_front_end.get_name() + "_frontend")
                moc_front_end.add_extension(TelemetryExtension("mo", t.send_event, t.send_error, t.send_stack_trace))
                moc_front_end.add_extension(ProgressReporterExtension(progress_printer(argv)))
                if legacy_transformations_config_used(argv):
                    raise Error('Legacy extensions are not supported for the new frontend')
                if legacy_extensions_used(argv):
                    raise Error('Legacy transformations configuration is not supported for the new frontend')
                if tensorflow_custom_operations_config_update_used(argv) and is_tf:
                    raise Error('TensorFlow custom operation config is not supported for the new frontend')
                if new_extensions_used(argv):
                    for extension in argv.extensions:
                        moc_front_end.add_extension(extension)
                ngraph_function = moc_pipeline(argv, moc_front_end)
                return graph, ngraph_function
            except OpConversionFailure as ex:
                # in some set of operations (TF1 While), we have to fallback to the Legacy TensorFlow Frontend
                # this is the second attempt for the fallback
                if not update_fallback_with_conversion_error(argv.use_new_frontend, is_tf, str(ex), fallback_reasons):
                    # re-throw exception for all frontends except TensorFlow FE
                    # and in case unexpected conversion failures
                    raise
            finally:
                # TODO: remove this workaround once new TensorFlow frontend supports non-frozen formats: checkpoint, MetaGraph, and SavedModel
                # Now it converts all TensorFlow formats to the frozen .pb format in case new TensorFlow frontend
                if is_tf and path_to_aux_pb is not None:
                    argv.input_model = orig_argv_values["input_model"]
                    argv.model_name = orig_argv_values["model_name"]
                    if os.path.exists(path_to_aux_pb):
                        os.remove(path_to_aux_pb)

    if len(fallback_reasons) > 0:
        reasons_message = ", ".join(fallback_reasons)
        load_extensions(argv, *list(deduce_legacy_frontend_by_namespace(argv)))
        t.send_event("mo", "fallback_reason", reasons_message)
        log.warning("The IR preparation was executed by the legacy MO path. "
                    "This is a fallback scenario applicable only for some specific cases. "
                    f"The detailed reason why fallback was executed: not supported {reasons_message} were used. "
                    "You can specify --use_new_frontend flag to force using the Frontend MO path to avoid additional checks. " +
                    refer_to_faq_msg(105))

    t.send_event("mo", "conversion_method", "mo_legacy")
    graph = unified_pipeline(argv)

    return graph, ngraph_function


def read_model(fem: FrontEndManager, path_to_xml: str):
    # We have to separate fe object lifetime from fem to
    # avoid segfault during object destruction. So fe must
    # be destructed before fem object explicitly.
    fe = fem.load_by_framework(framework="ir")
    function = fe.convert(fe.load(path_to_xml))
    return function


def emit_ir(graph: Graph, argv: argparse.Namespace, non_default_params: dict):
    NormalizeTI().find_and_replace_pattern(graph)
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    if 'feManager' in argv:
        del argv.feManager

    mean_data = deepcopy(graph.graph['mf']) if 'mf' in graph.graph else None
    input_names = deepcopy(graph.graph['input_names']) if 'input_names' in graph.graph else []

    output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
    orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))

    def clear_tmp_ir_files():
        for suf in [".xml", ".bin", ".mapping"]:
            # remove existing files
            path_to_file = orig_model_name + "_tmp" + suf
            if os.path.exists(path_to_file):
                os.remove(path_to_file)

    try:
        prepare_emit_ir(graph=graph,
                        data_type=graph.graph['cmd_params'].data_type,
                        output_dir=argv.output_dir,
                        output_model_name=argv.model_name,
                        mean_data=mean_data,
                        input_names=input_names,
                        meta_info=non_default_params,
                        use_temporary_path=True)

        fem = FrontEndManager()
        func = read_model(fem, orig_model_name + "_tmp.xml")
    except Exception as err:
        raise Error('Exception occurred while serialization or reading of the temporary IR: {}'.format(
            str(err),
        )) from err
    finally:
        # This graph cleanup is required to avoid double memory consumption
        graph.clear()
        clear_tmp_ir_files()

    return_code = "not executed"
    if not (argv.framework == 'tf' and argv.tensorflow_custom_operations_config_update):
        try:
            from openvino.tools.mo.back.offline_transformations import apply_offline_transformations
            func = apply_offline_transformations(func, argv)
            if "compress_fp16" in argv and argv.compress_fp16:
                # restore data_type cmd parameter
                argv.data_type = 'FP16'
            return_code = 0
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

    return func


def check_model_object(argv):
    model = argv['input_model']
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        from tensorflow.python.training.tracking.base import Trackable

        if isinstance(model, tf.compat.v1.GraphDef):
            return "tf"
        if isinstance(model, tf.compat.v1.Session):
            argv['input_model'] = model.graph_def
            return "tf"
        if isinstance(model, tf.types.experimental.ConcreteFunction):
            argv['input_model'] = model.graph.as_graph_def()
            return "tf"
        if isinstance(model, tf.keras.Model):
            return "tf"
        if isinstance(model, tf.train.Checkpoint):
            if isinstance(model.root, tf.keras.Model):
                argv['input_model'] = model.root
                return "tf"
            else:
                raise Error("Unknown checkpoint format.")

        if isinstance(model, tf.keras.layers.Layer) or isinstance(model, tf.Module):
            assert 'input_shape' in argv and argv['input_shape'] is not None, \
                "Converting of {} requires providing of input_shape.".format(type(model))
            assert len(argv['input_shape']) > 0, "Please provide non-empty input shape."
            inputs = []
            for shape_idx, shape in enumerate(parse_input_shapes(argv)):
                inp_shape = get_static_shape(shape)
                batch_size = None
                if len(inp_shape) > 1:
                    batch_size = inp_shape[0]
                    inp_shape = inp_shape[1:]
                inputs.append(tf.keras.Input(shape=inp_shape, batch_size=batch_size))
            outputs = model(*inputs)
            argv['input_model'] = tf.keras.Model(inputs, outputs)
            argv['input_shape'] = None
            return "tf"
        if isinstance(model, Trackable):
            return "tf"
    if 'torch' in sys.modules:
        import torch
        if isinstance(model, torch.nn.Module) or isinstance(model, torch.jit.ScriptFunction):
            return "pytorch"

    import io
    if isinstance(model, io.BytesIO):
        return 'onnx'

    raise Error('Unknown model type: {}'.format(type(model)))


def driver(argv: argparse.Namespace, non_default_params: dict):
    init_logger(argv.log_level.upper(), argv.silent)

    # Log dictionary with non-default cli parameters where complex classes are excluded.
    log.debug(str(non_default_params))

    start_time = datetime.datetime.now()

    graph, ngraph_function = prepare_ir(argv)
    legacy_path = False
    if graph is not None:
        res_ngraph_function = emit_ir(graph, argv, non_default_params)
        legacy_path = True
    else:
        res_ngraph_function = moc_emit_ir(ngraph_function, argv)

    if res_ngraph_function is None:
        return res_ngraph_function

    if not argv.silent:
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

    return res_ngraph_function, legacy_path


def args_dict_to_list(cli_parser, **kwargs):
    result = []
    for key, value in kwargs.items():
        if value is not None and cli_parser.get_default(key) != value:
            # skip parser checking for non str objects
            if not isinstance(value, str):
                continue
            result.append('--{}'.format(key))
            if not isinstance(value, bool):
                result.append(value)

    return result


def get_non_default_params(argv, cli_parser):
    import numbers
    # make dictionary with parameters which have non-default values to be serialized in IR in rt_info
    non_default_params = {}
    for arg, arg_value in vars(argv).items():
        if arg_value != cli_parser.get_default(arg):
            value = depersonalize(arg_value, arg)
            # Skip complex classes in params to prevent
            # serializing it to rt_info
            if isinstance(value, (str, bool, numbers.Number)):
                non_default_params[arg] = value
    return non_default_params


def params_to_string(**kwargs):
    all_params = {}
    for key, value in mo_convert_params.items():
        all_params.update(value)

    for key, value in kwargs.items():
        if key in all_params:
            param_data = all_params[key]
            if param_data.to_string is not None:
                kwargs[key] = param_data.to_string(value)
    return kwargs


def add_line_breaks(text: str, char_num: int, line_break: str):
    words = text.split(" ")
    cnt = 0
    for i, w in enumerate(words):
        cnt += len(w)
        if '\n' in w:
            cnt = len(w) - w.find('\n') - 1
        if cnt > char_num:
            if words[i][-1] not in ['\n', '\t']:
                words[i] = w + '\n'
            cnt = 0
    text = ' '.join(words).replace("\n ", "\n")
    return line_break + text.replace("\n", line_break)


def show_mo_convert_help():
    for group_name, group in mo_convert_params.items():
        if group_name == "optional":
            print("optional arguments:")
        elif group_name == "fw_agnostic":
            print("Framework-agnostic parameters:")
        elif group_name == "tf":
            print("TensorFlow*-specific parameters:")
        elif group_name == "caffe":
            print("Caffe*-specific parameters:")
        elif group_name == "mxnet":
            print("Mxnet-specific parameters:")
        elif group_name == "kaldi":
            print("Kaldi-specific parameters:")
        elif group_name == "pytorch":
            print("Pytorch-specific parameters:")
        else:
            raise Error("Unknown parameters group {}.".format(group_name))
        for param_name in group:
            param_data = group[param_name]
            text = param_data.description.format(param_data.possible_types_python_api)
            text = add_line_breaks(text, 56, "\n\t\t\t")
            print("  --{} {}".format(param_name, text))
        print()


def input_model_is_object(argv):
    # Input model can be set as object only for --input_model parameter.
    # --saved_model_dir or meta specific options are only used to store paths to the input model.
    if 'input_model' not in argv:
        return False
    if isinstance(argv['input_model'], (str, Path)):
        return False
    if argv['input_model'] is None:
        return False
    return True


def pack_params_to_args_namespace(args: dict, cli_parser: argparse.ArgumentParser):
    if len(args) > 0:
        args_string = params_to_string(**args)
        argv, _ = cli_parser.parse_known_args(args_dict_to_list(cli_parser, **args_string))

        # get list of all available params for convert_model()
        all_params = {}
        for key, value in mo_convert_params.items():
            all_params.update(value)

        # check that there are no unknown params provided
        for key, value in args_string.items():
            if key not in argv and key not in all_params.keys():
                raise Error("Unrecognized argument: {}".format(key))

            # Non string params like input_model or extensions are ignored by parse_args()
            # so we need to set them in argv separately
            if value is not None and getattr(argv, key, None) != value:
                setattr(argv, key, value)
    else:
        argv = cli_parser.parse_args()
    return argv


def _convert(cli_parser: argparse.ArgumentParser, framework, args):
    if 'help' in args and args['help']:
        show_mo_convert_help()
        return None, None

    telemetry = tm.Telemetry(tid=get_tid(), app_name='Model Optimizer', app_version=get_simplified_mo_version())
    telemetry.start_session('mo')
    telemetry.send_event('mo', 'version', get_simplified_mo_version())
    # Initialize logger with 'ERROR' as default level to be able to form nice messages
    # before arg parser deliver log_level requested by user
    init_logger('ERROR', False)
    try:
        model_framework = None
        inp_model_is_object = input_model_is_object(args)
        if inp_model_is_object:
            model_framework = check_model_object(args)
            if model_framework == "pytorch":
                example_inputs = None
                if 'example_input' in args and args['example_input'] is not None:
                    example_inputs = args['example_input']
                   
                if 'use_legacy_frontend' in args and args['use_legacy_frontend']:
                    # TO DO: remove this path, when pytorch frontend productization is finished, CVS-103726
                    # prevent invoking legacy mo python onnx frontend for models converted on the fly
                    args.pop("use_legacy_frontend")
                    return convert_pytorch_via_onnx(args, example_inputs, cli_parser, framework, _convert)

                decoder, input_signature  = get_pytorch_decoder(args['input_model'], parse_input_shapes(args), example_inputs)
                args['input_model'] = decoder
                args["framework"] = "pytorch"
                args["input_signature"] = input_signature

        argv = pack_params_to_args_namespace(args, cli_parser)

        if framework is not None:
            setattr(argv, 'framework', framework)

        # send telemetry with params info
        send_params_info(argv, cli_parser)

        non_default_params = get_non_default_params(argv, cli_parser)

        if inp_model_is_object:
            argv.model_name = "model"
        if not hasattr(argv, "model_name") or argv.model_name is None:
            argv.model_name = get_model_name_from_args(argv)

        if model_framework is not None:
            if argv.framework is not None:
                if argv.framework != model_framework:
                    raise Error("Provided model does not correspond to provided framework. The provided "
                                "framework is {}, the model type is {} which is expected to be {} framework.".format(
                                    argv.framework,
                                    type(argv.input_model),
                                    model_framework))
            else:
                argv.framework = model_framework

        argv.feManager = FrontEndManager()
        ov_model, legacy_path = driver(argv, {"conversion_parameters": non_default_params})

        # add MO meta data to model
        ov_model.set_rt_info(get_version(), "MO_version")
        ov_model.set_rt_info(get_rt_version(), "Runtime_version")
        ov_model.set_rt_info(str(legacy_path), "legacy_frontend")
        for key, value in non_default_params.items():
            ov_model.set_rt_info(str(value), ["conversion_parameters", str(key)])

        telemetry.send_event('mo', 'conversion_result', 'success')
        telemetry.end_session('mo')
        telemetry.force_shutdown(1.0)
        return ov_model, argv
    except Exception as e:
        telemetry.send_event('mo', 'conversion_result', 'fail')
        telemetry.end_session('mo')
        telemetry.force_shutdown(1.0)
        raise e.with_traceback(None)
