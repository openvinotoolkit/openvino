# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from copy import deepcopy

from e2e_tests.test_utils.reshape_tests_utils import get_mo_input_with_frozen_values, reorder_shapes_to_old_api, \
    get_input_data
from e2e_tests.test_utils.test_utils import prepare_data_consecutive_inferences


def mo_reshape_config(pipeline, shapes, instance_class_name):
    mo_config = deepcopy(pipeline)

    # we force model optimizer to generate already reshaped IR
    mo_shape = deepcopy(shapes)  # shapes: dict(name=list(shape_in_IE_layout))
    mo_arg_input = pipeline['get_ir']['get_ovc_model'].get('additional_args').get('input')

    if mo_arg_input is not None and "->" in mo_arg_input:
        cmd_mo_input = get_mo_input_with_frozen_values(mo_arg_input, shapes)
        mo_config['get_ir']['get_ovc_model']['additional_args'].update({'input': ','.join(list(map(str, cmd_mo_input)))})
    else:
        mo_config['get_ir']['get_ovc_model']['additional_args'].update({
            'input': ','.join(list(map(str, mo_shape.keys()))),
            'input_shape': ','.join(list(map(str, mo_shape.values()))),
        })
    # prevent MO reshape keys usage
    for attribute in ['batch']:
        if attribute in mo_config['get_ir']['get_ovc_model']['additional_args']:
            del mo_config['get_ir']['get_ovc_model']['additional_args'][attribute]
    # prevent IE network modifications
    infer_step_name = list(mo_config['infer'].keys())[0]
    if 'network_modifiers' in mo_config["infer"][infer_step_name]:
        del mo_config["infer"][infer_step_name]['network_modifiers']

    mo_config = update_pre_post_process_reshape_config(mo_config, shapes, instance_class_name)

    return mo_config


def ie_reshape_config(pipeline, shapes, test_name):
    ie_config = deepcopy(pipeline)
    ie_shapes = deepcopy(shapes)

    if test_name.lower().startswith('tf') and 'ie_sync' in pipeline['infer']:
        ie_shapes = reorder_shapes_to_old_api(shapes)

    # we force model optimizer to generate reshapable IR
    ie_config['infer'][list(ie_config['infer'].keys())[0]]['network_modifiers'] = {'reshape': {'shapes': ie_shapes}}
    ie_config = update_pre_post_process_reshape_config(ie_config, shapes, test_name)

    return ie_config


def update_pre_post_process_reshape_config(instance_ie_pipeline, shapes, instance_class_name, default_shapes=None,
                                           changed_values=None, layout=None, changed_dims=None,
                                           consecutive_infer=False):
    config = deepcopy(instance_ie_pipeline)
    ie_api = next(iter(config['infer']))

    # preprocess stage
    config['infer'][ie_api]['consecutive_infer'] = consecutive_infer
    if 'preprocess' not in config:
        stages = OrderedDict()
        for stage in config:
            stages[stage] = config[stage]
            if stage == 'read_input':
                stages['preprocess'] = OrderedDict()
        config.clear()
        config = stages
    # There is no need to reorder 'default_shapes' since we do not run old API for dynamism
    if instance_class_name.lower().startswith('tf') and 'ie_sync' in instance_ie_pipeline['infer']:
        shapes = reorder_shapes_to_old_api(shapes)
    if not consecutive_infer:
        config['preprocess'].update(get_input_data(shapes))
    else:
        config['preprocess'].update(prepare_data_consecutive_inferences(default_shapes, changed_values, layout,
                                                                        changed_dims))

    # postprocess stage
    if 'postprocessor' in instance_ie_pipeline:
        shape = iter(shapes.values()).__next__()
        for action_name, action_attrs in instance_ie_pipeline['postprocessor'].items():
            if 'batch' in action_attrs:
                action_attrs['batch'] = shape[0]
    return config


def dynamism_config(instance_ie_pipeline, shapes, test_name, default_shapes, changed_values, layout, changed_dims,
                    consecutive_infer_num):
    dynamic_config = deepcopy(instance_ie_pipeline)

    reshape_action_list = ['set_batch_using_reshape', 'reshape']
    infer_network_modifiers = {}
    if dynamic_config['infer'][list(dynamic_config['infer'].keys())[0]].get('network_modifiers'):
        for item in dynamic_config['infer'][list(dynamic_config['infer'].keys())[0]]['network_modifiers']:
            if item not in reshape_action_list:
                infer_network_modifiers[item] = \
                    dynamic_config['infer'][list(dynamic_config['infer'].keys())[0]]['network_modifiers'][item]

    dynamic_config['infer'][list(dynamic_config['infer'].keys())[0]]['network_modifiers'] = {
        'reshape': {'shapes': shapes}}
    dynamic_config['infer'][list(dynamic_config['infer'].keys())[0]]['network_modifiers'].update(
        infer_network_modifiers)
    if consecutive_infer_num:
        dynamic_config = update_pre_post_process_reshape_config(dynamic_config, shapes, test_name, default_shapes,
                                                                changed_values, layout, changed_dims,
                                                                consecutive_infer_num)

    return dynamic_config


def get_original_model_importer_pipeline_config(instance_ie_pipeline):
    """
    This function configures the pipeline which produces the results to be tested.
    In this pipeline the ONNX model is loaded into IE directly from a .onnx file without MO
    The network created from a model is then reshaped according to the configuration in 'shapes'
    """
    ie_config = deepcopy(instance_ie_pipeline)
    model_path = ie_config["get_ir"]["get_ovc_model"]["model"]

    # discard the IR generation with MO which comes from the original pipeline
    del ie_config["get_ir"]

    # reconfigure pipeline to use IE ONNX reader step instead of default IE step
    ie_api = next(iter(ie_config["infer"]))
    ie_config["infer"][ie_api]["model_path"] = model_path

    return ie_config
