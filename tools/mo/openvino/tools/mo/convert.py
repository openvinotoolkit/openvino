import argparse
import datetime
import logging as log
import os
import platform
import sys
import traceback
from collections import OrderedDict, namedtuple
from copy import deepcopy
from pathlib import Path
import numpy as np

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

from openvino.tools.mo.back.SpecialNodesFinalization import RemoveConstOps, CreateConstNodesReplacement, NormalizeTI
from openvino.tools.mo.moc_frontend.check_config import legacy_extensions_used, legacy_transformations_config_used, \
    new_extensions_used, new_transformations_config_used, input_freezig_used
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
    get_mean_scale_dictionary, get_meta_info, get_mxnet_cli_options, get_onnx_cli_options, \
    get_placeholder_shapes, get_tf_cli_options, get_tuple_values, parse_transform, parse_tuple_pairs, get_all_cli_parser
from openvino.tools.mo.utils.error import Error, FrameworkError
from openvino.tools.mo.utils.find_ie_version import find_ie_version
from openvino.tools.mo.utils.get_ov_update_message import get_ov_update_message, get_ov_api20_message
from openvino.tools.mo.utils.guess_framework import deduce_legacy_frontend_by_namespace
from openvino.tools.mo.utils.logger import init_logger, progress_printer
from openvino.tools.mo.utils.model_analysis import AnalysisResults
from openvino.tools.mo.utils.utils import refer_to_faq_msg
from openvino.tools.mo.utils.telemetry_utils import send_params_info, send_framework_info
from openvino.tools.mo.utils.version import get_simplified_mo_version, get_simplified_ie_version
from openvino.tools.mo.utils.versions_checker import check_requirements  # pylint: disable=no-name-in-module
from openvino.tools.mo.utils.telemetry_utils import get_tid
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.runtime import Layout, PartialShape, Dimension, Shape, Type
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager, ProgressReporterExtension, TelemetryExtension, JsonConfigExtension

InputCutInfo = namedtuple("InputInfo", ["name", "shape", "type", "value"], defaults=[None, None, None])
LayoutMap = namedtuple("LayoutMap", ["source_layout", "target_layout"], defaults=[None, None])


def load_extensions(argv: argparse.Namespace, is_tf: bool, is_caffe: bool, is_mxnet: bool, is_kaldi: bool,
                    is_onnx: bool):
    extensions = None
    if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
        extensions = argv.extensions.split(',')
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
            raise Error('Framework {} is not a valid target. Please use --framework with one from the list: {}. ' +
                        refer_to_faq_msg(15), argv.framework, frameworks)

    if is_legacy_frontend:
        if new_extensions_used(argv):
            raise Error('New kind of extensions used on legacy path')
        if new_transformations_config_used(argv):
            raise Error('New kind of transformations configuration used on legacy path')

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
                    "try to install OpenVINO (TM) Toolkit using \"install_prerequisites.{}\"".format(
            "bat" if sys.platform == "windows" else "sh"))

    try:
        if not find_ie_version(silent=argv.silent):
            raise_ie_not_found()
    except Exception as e:
        log.error(e)
        raise_ie_not_found()

    if 'data_type' in argv and argv.data_type in ['FP16', 'half']:
        argv.data_type = 'FP32'
        argv.compress_fp16 = True
    else:
        argv.compress_fp16 = False

    # This is just to check that transform key is valid and transformations are available
    check_available_transforms(parse_transform(argv.transform))

    # For C++ frontends there are no specific Python installation requirements, check only generic ones
    if moc_front_end:
        ret_code = check_requirements()
    else:
        ret_code = check_requirements(framework=argv.framework)
    if ret_code:
        raise Error('check_requirements exited with return code {}'.format(ret_code))

    if is_tf and argv.tensorflow_use_custom_operations_config is not None:
        argv.transformations_config = argv.tensorflow_use_custom_operations_config

    if is_caffe and argv.mean_file and argv.mean_values:
        raise Error('Both --mean_file and mean_values are specified. Specify either mean file or mean values. ' +
                    refer_to_faq_msg(17))
    elif is_caffe and argv.mean_file and argv.mean_file_offsets:
        values = get_tuple_values(argv.mean_file_offsets, t=int, num_exp_values=2)
        mean_file_offsets = mo_array([int(x) for x in values[0].split(',')])
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

    # There is no possibility for fallback if a user strictly wants to use new frontend
    if argv.use_new_frontend:
        return fallback_reasons

    fallback_reasons['extensions'] = legacy_extensions_used
    fallback_reasons['transformations_config'] = legacy_transformations_config_used
    fallback_reasons['input_freezing'] = input_freezig_used

    reasons = [reason for reason, is_applicable in fallback_reasons.items() if is_applicable(argv)]
    return reasons


def get_default_frontends():
    # Set which frontend to use by default, values should be 'new' or 'legacy'
    default_frontends = {
        'onnx': 'new',
        'tf': 'legacy'
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
    argv = arguments_post_parsing(argv)
    t = tm.Telemetry()
    graph = None
    ngraph_function = None
    moc_front_end, available_moc_front_ends = get_moc_frontends(argv)
    if moc_front_end:
        fallback_reasons = check_fallback(argv)
        if len(fallback_reasons) == 0:
            t.send_event("mo", "conversion_method", moc_front_end.get_name() + "_frontend")
            moc_front_end.add_extension(TelemetryExtension("mo", t.send_event, t.send_error, t.send_stack_trace))
            moc_front_end.add_extension(ProgressReporterExtension(progress_printer(argv)))
            if legacy_transformations_config_used(argv):
                raise Error('Legacy extensions are not supported for the new frontend')
            if legacy_extensions_used(argv):
                raise Error('Legacy transformations configuration is not supported for the new frontend')
            if new_transformations_config_used(argv):
                moc_front_end.add_extension(JsonConfigExtension(argv.transformations_config))
            if new_extensions_used(argv):
                for extension in argv.extensions.split(','):
                    moc_front_end.add_extension(extension)
            ngraph_function = moc_pipeline(argv, moc_front_end)
            return graph, ngraph_function
        else:  # apply fallback
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


def emit_ir(graph: Graph, argv: argparse.Namespace):
    NormalizeTI().find_and_replace_pattern(graph)
    for_graph_and_each_sub_graph_recursively(graph, RemoveConstOps().find_and_replace_pattern)
    for_graph_and_each_sub_graph_recursively(graph, CreateConstNodesReplacement().find_and_replace_pattern)

    if 'feManager' in argv:
        del argv.feManager

    mean_data = deepcopy(graph.graph['mf']) if 'mf' in graph.graph else None
    input_names = deepcopy(graph.graph['input_names']) if 'input_names' in graph.graph else []
    # needed for tmp IR generation
    output_dir = os.getcwd()

    prepare_emit_ir(graph=graph,
                    data_type=graph.graph['cmd_params'].data_type,
                    output_dir=output_dir,
                    output_model_name=argv.model_name,
                    mean_data=mean_data,
                    input_names=input_names,
                    meta_info=get_meta_info(argv),
                    use_temporary_path=True)

    # This graph cleanup is required to avoid double memory consumption
    graph.clear()

    if not (argv.framework == 'tf' and argv.tensorflow_custom_operations_config_update):
        # needed for tmp IR generation
        output_dir = os.getcwd()
        orig_model_name = os.path.normpath(os.path.join(output_dir, argv.model_name))

        return_code = "not executed"
        try:
            from openvino.tools.mo.back.offline_transformations import apply_offline_transformations
            func = apply_offline_transformations(orig_model_name, argv)
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

        for suf in [".xml", ".bin", ".mapping"]:
            # remove existing files
            path_to_file = orig_model_name + "_tmp" + suf
            if os.path.exists(path_to_file):
                os.remove(path_to_file)
    return func


def driver(argv: argparse.Namespace):
    init_logger(argv.log_level.upper(), argv.silent)

    start_time = datetime.datetime.now()

    graph, ngraph_function = prepare_ir(argv)
    if graph is not None:
        res_ngraph_function = emit_ir(graph, argv)
    else:
        res_ngraph_function = moc_emit_ir(ngraph_function, argv)

    if res_ngraph_function is None:
        return res_ngraph_function

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

    return res_ngraph_function


def pack_params_to_args_namespace(**kwargs):
    fe_manager = FrontEndManager()
    cli_parser = get_all_cli_parser(fe_manager)
    argv = cli_parser.parse_args()
    for key, value in kwargs.items():
        if value is not None:
            setattr(argv, key, value)
    send_params_info(argv, cli_parser)
    return argv


def path_to_str(path):
    if path is None:
        return None
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        return str(path)
    else:
        raise Exception("Incorrect type of {} expected str or Path, got {}".format(path, type(path)))


def paths_to_str(paths):
    if paths is None:
        return None
    if isinstance(paths, list):
        paths_str = []
        for path in paths:
            paths_str.append(path_to_str(path))
        return ','.join(paths_str)
    else:
        path_to_str(paths)


def str_list_to_str(values):
    if values is None:
        return None
    if isinstance(values, str):
        return values
    elif isinstance(values, list):
        values_str = []
        for value in values:
            if not isinstance(value, str):
                raise Error("Incorrect argument. {} expected to string, got type {}.".format(value, type(value)))
            values_str.append(value)
        return ','.join(values_str)
    else:
        Error("Incorrect argument. {} expected to string or list of strings, got type {}.".format(values, type(values)))


def extension_path_to_str_or_extensions_class(extension):
    if isinstance(extension, str):
        return extension
    elif isinstance(extension, Path):
        return str(extension)
    else:
        #TODO: Add support of Extension objects
        raise Exception("Currently usage of Extension classes is not supported.")


def extensions_to_str_or_extensions_class(extensions):
    if extensions is None:
        return None
    if isinstance(extensions, list):
        ext_list = []
        for ext in extensions:
            ext = extension_path_to_str_or_extensions_class(ext)
            ext_list.append(ext)
        return ','.join(ext_list)
    else:
        return extension_path_to_str_or_extensions_class(extensions)


def dimension_to_str(dim: Dimension):
    # TODO: replace this code with Dimension to string conversion method from openvino.runtime when 69092 is done
    if dim.is_static:
        return str(dim.get_length())
    if dim.get_min_length() > 0:
        dim_str = str(dim.get_min_length()) + ".."
        if dim.get_max_length() < np.iinfo(np.int64).max:
            dim_str += str(dim.get_max_length())
    elif dim.get_max_length() < np.iinfo(np.int64).max:
        return ".." + str(dim.get_min_length())
    else:
        return "?"


def partial_shape_to_str(shape: PartialShape, separator: str):
    # TODO: replace this code with PartialShape to string conversion method from openvino.runtime when 69092 is done

    dims = []
    for i in range(shape.rank.get_length()):
        dims.append(dimension_to_str(shape.get_dimension(i)))

    return "[" + separator.join(dims) + "]"


def shape_to_str(shape, separator):
    if isinstance(shape, str):
        return shape
    if isinstance(shape, PartialShape):
        return partial_shape_to_str(shape, separator)
    if isinstance(shape, Shape):
        return partial_shape_to_str(PartialShape(shape), separator)
    if isinstance(shape, list):
        dims = []
        for dim in shape:
            if isinstance(dim, Dimension):
                dims.append(dimension_to_str(dim))
            elif isinstance(dim, int):
                dims.append(str(dim))
            else:
                raise Exception("Incorrect type of dimension. Expected Dimension or int, got {}".format(type(dim)))
        return "[" + separator.join(dims) + "]"
    raise Exception("Incorrect shape type. Expected PartialShape, Shape, [Dimension, ...] or [int, ...], "
                    "got {}".format(type(shape)))


def input_shape_to_str(input_shape):
    if input_shape is None or isinstance(input_shape, str):
        return input_shape
    if isinstance(input_shape, list):
        shapes = []
        for shape in input_shape:
            shapes.append(shape_to_str(shape, ','))
        return ','.join(shapes)
    return shape_to_str(input_shape, ',')


def type_to_str(type_obj):
    if isinstance(type_obj, str):
        return type_obj
    if isinstance(type_obj, type):
        return np_data_type_to_destination_type(type_obj)
    if isinstance(type_obj, Type):
        return type_obj.get_type_name()
    raise Exception("Incorrect type. Expected Type or numpy type, got {}".format(type(type_obj)))


def value_to_str(value, separator):
    if isinstance(value, np.ndarray):
        values = []
        for x in np.nditer(value):
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    if isinstance(value, list):
        values = []
        for x in value:
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    raise Exception("Incorrect value type. Expected np.ndarray or list, got {}".format(type(value)))


def single_input_to_str(input):
    if isinstance(input, str):
        return input
    if isinstance(input, InputCutInfo):
        if not isinstance(input.name, str):
            raise Exception("Input name should be string, got {}".format(input.name))
        input_str = input.name
        if input.shape is not None:
            input_str += shape_to_str(input.shape, " ")
        if input.type is not None:
            input_str += "{" + type_to_str(input.type) + "}"
        if input.value is not None:
            input_str += "->" + value_to_str(input.value, " ")
        return input_str


def input_to_str(input):
    if input is None or isinstance(input, str):
        return input
    if isinstance(input, list):
        inputs_str = []
        for inp in input:
            inputs_str.append(single_input_to_str(inp))
        return ','.join(inputs_str)
    return single_input_to_str(input)


def mean_scale_value_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, val in value.values():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + value_to_str(val, ","))
        return ",".join(values_str)
    return value_to_str(value, ",")


def layout_to_str(layout):
    if isinstance(layout, str):
        return layout
    if isinstance(layout, Layout):
        return layout.to_string()
    raise Exception("Incorrect layout type. Expected Layout or string, got {}".format(type(layout)))


def source_target_layout_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.values():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layout_to_str(layout)) + ")"
        return ",".join(values_str)

    raise Exception("Incorrect layout. Expected dictionary, where key is operation name and value is Layout. Got {}".format(value))


def layoutmap_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, LayoutMap):
        source_layout = None
        target_layout = None
        if value.source_layout is not None:
            source_layout = layout_to_str(value.source_layout)
        if value.target_layout is not None:
            target_layout = layout_to_str(value.target_layout)
        if source_layout is not None and target_layout is not None:
            return source_layout + "->" + target_layout
        if source_layout is not None:
            return source_layout
        elif target_layout is not None:
            return target_layout
        else:
            raise Exception("Incorrect LayoutMap, source or target layout should be specified.")
    return layout_to_str(value)


def layout_param_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.values():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layoutmap_to_str(layout)) + ")"
        return ",".join(values_str)

    raise Exception("Incorrect layout. Expected dictionary, where key is operation name and value is "
                    "Layout or LayoutMap. Got {}".format(value))


def convert(
        input_model,
        framework=None,
        model_name=None,
        input_shape=None,
        scale=None,
        reverse_input_channels=None,
        log_level=None,
        input=None,
        output=None,
        mean_values=(),
        scale_values=(),
        source_layout=(),
        target_layout=(),
        layout=(),
        transform=None,
        extensions=None,
        batch=None,
        silent=None,
        static_shape=None,
        progress=None,
        stream_output=None,
        transformations_config=None,
        use_new_frontend=None,
        use_legacy_frontend=None,
        disable_omitting_optional=None,
        enable_flattening_nested_params=None,
        input_model_is_text=None,
        input_checkpoint=None,
        input_meta_graph=None,
        saved_model_dir=None,
        saved_model_tags=None,
        tensorflow_custom_operations_config_update=None,
        tensorflow_object_detection_api_pipeline_config=None,
        tensorboard_logdir=None,
        tensorflow_custom_layer_libraries=None,
        input_proto=None,
        caffe_parser_path=None,
        k=None,
        input_symbol=None,
        nd_prefix_name=None,
        pretrained_model_name=None,
        save_params_from_nd=None,
        legacy_mxnet_model=None,
        enable_ssd_gluoncv=None,
        counts=None,
        remove_output_softmax=None,
        remove_memory=None
):
    telemetry = tm.Telemetry(tid=get_tid(), app_name='Model Optimizer', app_version=get_simplified_mo_version())
    telemetry.start_session('mo')
    telemetry.send_event('mo', 'version', get_simplified_mo_version())
    argv = pack_params_to_args_namespace(
        input_model=path_to_str(input_model),
        framework=framework,
        model_name=model_name,
        input_shape=input_shape_to_str(input_shape),
        scale=scale,
        reverse_input_channels=reverse_input_channels,
        log_level=log_level,
        input=input_to_str(input),
        output=str_list_to_str(output),
        mean_values=mean_scale_value_to_str(mean_values),
        scale_values=mean_scale_value_to_str(scale_values),
        source_layout=source_target_layout_to_str(source_layout),
        target_layout=source_target_layout_to_str(target_layout),
        layout=layout_param_to_str(layout),
        transform=transform,
        extensions=extensions_to_str_or_extensions_class(extensions),
        batch=batch,
        silent=silent,
        static_shape=static_shape,
        progress=progress,
        stream_output=stream_output,
        transformations_config=path_to_str(transformations_config),
        use_new_frontend=use_new_frontend,
        use_legacy_frontend=use_legacy_frontend,
        disable_omitting_optional=disable_omitting_optional,
        enable_flattening_nested_params=enable_flattening_nested_params,
        input_model_is_text=input_model_is_text,
        input_checkpoint=path_to_str(input_checkpoint),
        input_meta_graph=path_to_str(input_meta_graph),
        saved_model_dir=path_to_str(saved_model_dir),
        saved_model_tags=str_list_to_str(saved_model_tags),
        tensorflow_custom_operations_config_update=path_to_str(tensorflow_custom_operations_config_update),
        tensorflow_object_detection_api_pipeline_config=path_to_str(tensorflow_object_detection_api_pipeline_config),
        tensorboard_logdir=path_to_str(tensorboard_logdir),
        tensorflow_custom_layer_libraries=paths_to_str(tensorflow_custom_layer_libraries),
        input_proto=path_to_str(input_proto),
        caffe_parser_path=path_to_str(caffe_parser_path),
        k=path_to_str(k),
        input_symbol=path_to_str(input_symbol),
        nd_prefix_name=nd_prefix_name,
        pretrained_model_name=pretrained_model_name,
        save_params_from_nd=save_params_from_nd,
        legacy_mxnet_model=legacy_mxnet_model,
        enable_ssd_gluoncv=enable_ssd_gluoncv,
        counts=path_to_str(counts),
        remove_output_softmax=remove_output_softmax,
        remove_memory=remove_memory)
    try:
        # Initialize logger with 'ERROR' as default level to be able to form nice messages
        # before arg parser deliver log_level requested by user
        init_logger('ERROR', False)

        if framework:
            argv.framework = framework
        argv.feManager = FrontEndManager()

        ov_update_message = None
        ov_api20_message = None
        if not hasattr(argv, 'silent') or not argv.silent:
            ov_update_message = get_ov_update_message()
            ov_api20_message = get_ov_api20_message()
        ngraph_function = driver(argv)
        if ov_update_message:
            print(ov_update_message)
        if ov_api20_message and ngraph_function is not None:
            print(ov_api20_message)
        telemetry.send_event('mo', 'conversion_result', 'success')
        telemetry.end_session('mo')
        telemetry.force_shutdown(1.0)
        return ngraph_function
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
    return None
