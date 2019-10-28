"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import datetime
import logging as log
import os
import sys
import traceback
from collections import OrderedDict

import numpy as np

from mo.utils import import_extensions
from mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_model_name, \
    get_common_cli_options, get_caffe_cli_options, get_tf_cli_options, get_mxnet_cli_options, get_kaldi_cli_options, \
    get_onnx_cli_options, get_mean_scale_dictionary, parse_tuple_pairs, get_freeze_placeholder_values, \
    append_exp_keys_to_namespace
from mo.utils.error import Error, FrameworkError
from mo.utils.guess_framework import guess_framework_by_ext
from mo.utils.logger import init_logger
from mo.utils.utils import refer_to_faq_msg
from mo.utils.version import get_version
from mo.utils.versions_checker import check_requirements


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
                if op is 'k':
                    default_path = os.path.join(os.path.dirname(sys.argv[0]),
                                                'extensions/front/caffe/CustomLayersMapping.xml')
                    if getattr(argv, op, 'NONE') == default_path:
                        lines.append('\t{}: \t{}'.format(desc, 'Default'))
                        continue
                lines.append('\t{}: \t{}'.format(desc, getattr(argv, op, 'NONE')))
    lines.append('Model Optimizer version: \t{}'.format(get_version()))
    print('\n'.join(lines))


def driver(argv: argparse.Namespace):
    if argv.version:
        print('Version of Model Optimizer is: {}'.format(get_version()))
        return 0

    init_logger(argv.log_level.upper(), argv.silent)
    start_time = datetime.datetime.now()

    if not argv.framework:
        if 'saved_model_dir' in argv and argv.saved_model_dir or \
                'input_meta_graph' in argv and argv.input_meta_graph:
            argv.framework = 'tf'
        elif 'input_symbol ' in argv and argv.input_symbol or \
                'pretrained_model_name' in argv and argv.pretrained_model_name:
            argv.framework = 'mxnet'
        elif 'input_proto' in argv and argv.input_proto:
            argv.framework = 'caffe'
        elif argv.input_model is None:
            raise Error('Path to input model is required: use --input_model.')
        else:
            argv.framework = guess_framework_by_ext(argv.input_model)
        if not argv.framework:
            raise Error(
                'Framework name can not be deduced from the given options: {}={}. ' +
                'Use --framework to choose one of caffe, tf, mxnet, kaldi, onnx',
                '--input_model',
                argv.input_model,
                refer_to_faq_msg(15),
            )

    is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx = (argv.framework == x for x in
                                                    ['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'])

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

    log.debug('Output model name would be {}{{.xml, .bin}}'.format(model_name))

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
        print_argv(argv, is_caffe, is_tf, is_mxnet, is_kaldi, is_onnx, model_name)

    if not any([is_tf, is_caffe, is_mxnet, is_kaldi, is_onnx]):
        raise Error(
            'Framework {} is not a valid target. ' +
            'Please use --framework with one from the list: caffe, tf, mxnet, kaldi, onnx. ' +
            refer_to_faq_msg(15),
            argv.framework
        )

    ret_code = check_requirements(framework=argv.framework)
    if ret_code:
        return ret_code

    mean_file_offsets = None
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
    custom_layers_mapping_path = argv.k if is_caffe and argv.k else None

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

    argv.placeholder_shapes = get_placeholder_shapes(argv.input, argv.input_shape, argv.batch)

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

    ret_res = 1
    if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
        extensions = argv.extensions.split(',')
    else:
        extensions = None

    argv.freeze_placeholder_with_value, argv.input = get_freeze_placeholder_values(argv.input,
                                                                                   argv.freeze_placeholder_with_value)

    if is_tf:
        import mo.pipeline.tf as mo_tf
        from mo.front.tf.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        ret_res = mo_tf.tf2nx(argv, argv.input_model, model_name, argv.output_dir,
                              is_binary=not argv.input_model_is_text)

    elif is_caffe:
        import mo.pipeline.caffe as mo_caffe
        from mo.front.caffe.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        ret_res = mo_caffe.driver(argv, argv.input_proto, argv.input_model, model_name, argv.output_dir,
                                  argv.caffe_parser_path,
                                  mean_file=argv.mean_file,
                                  mean_file_offsets=mean_file_offsets,
                                  custom_layers_mapping_path=custom_layers_mapping_path)

    elif is_mxnet:
        import mo.pipeline.mx as mo_mxnet
        from mo.front.mxnet.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        ret_res = mo_mxnet.driver(argv, argv.input_model, model_name, argv.output_dir)

    elif is_kaldi:
        import mo.pipeline.kaldi as mo_kaldi
        from mo.front.kaldi.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        ret_res = mo_kaldi.driver(argv, argv.input_model, model_name, argv.output_dir)
    elif is_onnx:
        import mo.pipeline.onnx as mo_onnx
        from mo.front.onnx.register_custom_ops import get_front_classes
        import_extensions.load_dirs(argv.framework, extensions, get_front_classes)
        ret_res = mo_onnx.driver(argv, argv.input_model, model_name, argv.output_dir)

    if ret_res != 0:
        return ret_res
    if not (is_tf and argv.tensorflow_custom_operations_config_update) and not argv.silent:
        output_dir = argv.output_dir if argv.output_dir != '.' else os.getcwd()
        print('\n[ SUCCESS ] Generated IR model.')
        print('[ SUCCESS ] XML file: {}.xml'.format(os.path.join(output_dir, model_name)))
        print('[ SUCCESS ] BIN file: {}.bin'.format(os.path.join(output_dir, model_name)))
        elapsed_time = datetime.datetime.now() - start_time
        print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))
    return ret_res


def main(cli_parser: argparse.ArgumentParser, framework: str):
    try:
        # Initialize logger with 'ERROR' as default level to be able to form nice messages
        # before arg parser deliver log_level requested by user
        init_logger('ERROR', False)

        argv = cli_parser.parse_args()
        if framework:
            argv.framework = framework
        append_exp_keys_to_namespace(argv)
        return driver(argv)
    except (FileNotFoundError, NotADirectoryError) as e:
        log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
        log.debug(traceback.format_exc())
    except Error as err:
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
    return 1
