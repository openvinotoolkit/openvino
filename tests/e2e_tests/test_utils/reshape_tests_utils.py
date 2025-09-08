# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as log
import re
from collections import defaultdict
from itertools import zip_longest
from types import SimpleNamespace

import numpy as np

from e2e_tests.common.test_utils import name_aligner
from e2e_tests.pipelines.pipeline_templates.comparators_template import dummy_comparators, eltwise_comparators
from e2e_tests.common.comparator.container import ComparatorsContainer


def should_run_reshape(instance) -> bool:
    if not hasattr(instance, 'ie_pipeline'):
        # test does not involve IE
        return False

    if 'infer' not in instance.ie_pipeline:
        # test does not involve Infer step
        return False

    if 'get_ovc_model' not in instance.ie_pipeline['get_ir']:
        # can not reshape without `mo`
        return False

    if hasattr(instance, 'model_info') and instance.model_info.framework != 'dldt':
        # downloader models with IRs only are not tested by reshape
        return False

    if not hasattr(instance, 'input_descriptor'):
        # no info for reshape was provided
        log.info('Please, specify input_descriptor attribute for {}'.format(instance))
        return False

    if all([v.get('changeable_dims') is None for v in instance.input_descriptor.values()]):
        # model was set as non-reshape-able
        return False

    return True


def get_reshape_pipeline_pairs(instance) -> list:
    supported_pipelines = ['MO', 'IE']
    types = getattr(instance, 'requested_reshape_types', supported_pipelines)

    if len(types) == 1 or isinstance(types, str):
        log.info(f'Only {types} reshape pipeline was set for {instance.__class__.__name__}')
        return [types]
    else:
        pipelines_pairs = []
        for pipeline in types[1:]:
            pipelines_pairs.append([types[0], pipeline])
    return pipelines_pairs


def check_config(default_shapes, layout, changeable_dims):
    for input_layer, layer_value in changeable_dims.items():

        assert len(layout[input_layer]) == len(default_shapes[input_layer]), \
            'Layout {} and default_shapes {} of layer "{}"' \
            ' must have the same number of values'.format(
                layout[input_layer], default_shapes[input_layer], input_layer)

        if layer_value is not None:
            for dimension in layer_value:
                assert dimension in layout[input_layer], \
                    "Dimension '{}' wasn't found in input '{}'" \
                    " layout: {}".format(dimension, input_layer, layout[input_layer])

                for data in layer_value[dimension]:
                    assert len(data) == len(dimension), \
                        'Number of values {} for dimension "{}" in input layer "{}" should be the same' \
                        " as length of changeable dimension".format(data, dimension, input_layer)


def get_dims_to_change(changeable_dims):
    dims_to_change = defaultdict(list)
    for layer_name in changeable_dims:
        if changeable_dims[layer_name] is None:
            dims_to_change[layer_name].append(None)
        else:
            for dim in changeable_dims[layer_name]:
                for _ in range(len(changeable_dims[layer_name][dim])):
                    dims_to_change[layer_name].append(dim)

    dims_to_change = refactor_values(dims_to_change)

    return dims_to_change


def get_values_to_change(changeable_dims):
    values_to_change = defaultdict(list)
    for layer_name in changeable_dims:
        if changeable_dims[layer_name] is None:
            values_to_change[layer_name].append(None)
        else:
            for dim in changeable_dims[layer_name]:
                for value in changeable_dims[layer_name][dim]:
                    values_to_change[layer_name].append(value)

    return values_to_change


def refactor_values(values):
    refactored_values = lambda x: list(zip_longest(*x.values()))
    return [tuple(zip(values.keys(), obj)) for obj in refactored_values(values)]


def construct_new_shapes(test_number, default_shapes, layout, dims_to_change, values_to_change, dynamism_type=False):
    reshape_config = {k: list(v) for k, v in default_shapes.items()}
    for input_layer, input_dimension in dims_to_change[test_number]:
        if input_dimension is None:
            reshape_config[input_layer] = default_shapes[input_layer]
        else:
            # list comprehension is necessary in cases when multiply dimensions was set like 'HW'
            dim_indexes = [layout[input_layer].index(d) for d in input_dimension]
            # we should use default values if None is set as value
            for value_index, value in enumerate(values_to_change[input_layer][test_number]):
                if value is None:
                    continue
                else:
                    if not dynamism_type or dynamism_type == 'None':
                        reshape_config[input_layer][dim_indexes[value_index]] = value
                    if dynamism_type == 'negative_ones':
                        reshape_config[input_layer][dim_indexes[value_index]] = -1
                    if dynamism_type == 'range_values':
                        reshape_config[input_layer][dim_indexes[value_index]] = sorted([
                            default_shapes[input_layer][dim_indexes[value_index]], value])

    return reshape_config


def get_reshape_configurations(reshape_test_case, dynamism_type) -> list:
    """
    This function returns list of reshape configurations.

    Reshape configuration here is a list with info for reshape.
    It has the following structure:
    1. shapes: {input_layer: [input_layer_shapes], next_input_layer: [next_input_layer_shapes]}
    2. dimensions are supposed to be changed: {input_layer: dimension, next_input_layer: dimension}
    3. layout: layout of each input layer
    4. default shapes: dictionary with input layer names and its shapes
    """
    input_descriptor = reshape_test_case.input_descriptor

    default_shapes = {k: v['default_shape'] for k, v in input_descriptor.items() if not v.get('frozen_input')}
    layout = {k: v['layout'] for k, v in input_descriptor.items() if not v.get('frozen_input')}
    changeable_dims = {k: v['changeable_dims'] for k, v in input_descriptor.items() if not v.get('frozen_input')}
    check_config(default_shapes, layout, changeable_dims)

    reshape_configurations = []

    # get input layer-changed dimensions pairs for each specified shape value
    dims_to_change = get_dims_to_change(changeable_dims)
    # construct matrix of shape values for each input layer
    values_to_change = get_values_to_change(changeable_dims)
    # number of tests is number of input layer-changed dimensions pairs
    number_of_tests = len(dims_to_change)
    refactored_values = refactor_values(values_to_change)

    # construct new shapes from layer-changed dimensions pairs and matrix of values
    for test in range(number_of_tests):
        reshape_config = construct_new_shapes(test, default_shapes, layout, dims_to_change,
                                              values_to_change, dynamism_type)
        reshape_configurations.append(SimpleNamespace(shapes=reshape_config, changed_dims=dict(dims_to_change[test]),
                                                      layout=layout, default_shapes=default_shapes,
                                                      changed_values=refactored_values[test]))

    return reshape_configurations


def get_input_data(shapes):
    return {'dynamism_preproc': {'execution_function': lambda data: replicator(data, shapes)}}


def batch_was_changed(shapes, changed_dims, layout, default_shapes):
    batch = None

    for layer, dimension in changed_dims.items():
        if dimension is None:
            continue
        if len(dimension) > 1:
            continue
        index = layout[layer].index(dimension)
        # we assume that batch index is always == 0
        if index != 0:
            continue
        if shapes[layer][index] != default_shapes[layer][index]:
            batch = shapes[layer][index]

    return batch


def compare(instance, ref_results, cur_results):
    assert len(instance.comparators) == 1 or "dummy" not in instance.comparators, \
        "Dummy comparator is not the only one in comparators of instance"

    if not ref_results:
        ref_results = {}
        instance.comparators = dummy_comparators()
    else:
        instance.comparators = eltwise_comparators(device=getattr(instance, 'device'),
                                                   precision=getattr(instance, 'precision'),
                                                   a_eps=getattr(instance, 'a_eps', None),
                                                   r_eps=getattr(instance, 'r_eps', None))

    cur_results = cur_results.fetch_results()
    cur_results = cur_results if type(cur_results) is list else [cur_results]
    statuses = []
    for ref_result, cur_result in zip(ref_results, cur_results):
        comparators = ComparatorsContainer(
            config=instance.comparators,
            infer_result=cur_result,
            reference=ref_result.fetch_results(),
            result_aligner=name_aligner,
        )

        log.info('Running comparators:')
        comparators.apply_postprocessors()
        comparators.apply_all()
        statuses.append(comparators.report_statuses())

    return all(statuses)


def reorder_shapes_to_old_api(shapes):
    reorder_shapes = copy.deepcopy(shapes)
    for k, v in shapes.items():
        if len(v) in [4, 5]:
            reorder_shapes[k] = tuple(np.array(v).take((0, len(v) - 1, *list(range(1, len(v) - 1)))))
    return reorder_shapes


def get_mo_input_with_frozen_values(mo_arg_input, shapes):
    cmd_mo_input = []
    for input in mo_arg_input.split(","):
        if "->" in input:
            cmd_mo_input.append(input)
        else:
            input = re.sub(r"[(\[]([0-9  -]*)[)\]]", "", input)
            cmd_mo_input.append(input + str(shapes[input]).replace(',', ''))
    return cmd_mo_input


def prepare_data_consecutive_inferences(default_shapes, changed_values, layout, dims_to_change):
    def construct_input_data(data):
        input_data = copy.deepcopy(data)
        consecutive_infer_input_data = [data]

        changed_data_shapes = get_static_shape(default_shapes, changed_values, layout, dims_to_change)
        second_data = replicator(input_data, changed_data_shapes)
        consecutive_infer_input_data.append(second_data)

        return consecutive_infer_input_data
    return {'dynamism_preproc': {'execution_function': lambda data: construct_input_data(data)}}


def get_static_shape(default_shapes, changed_values, layout, dims_to_change):
    static_shapes = copy.deepcopy(default_shapes)
    static_shapes = {k: list(v) for k, v in static_shapes.items()}
    for input_layer, dimension in dims_to_change.items():
        if dimension is None:
            continue
        else:
            dim_indexes = [layout[input_layer].index(d) for d in dimension]
            for value_index, value in enumerate(dict(changed_values)[input_layer]):
                if value is None:
                    static_shapes[input_layer][dim_indexes[value_index]] = \
                        default_shapes[input_layer][dim_indexes[value_index]]
                else:
                    static_shapes[input_layer][dim_indexes[value_index]] = value
    return static_shapes


def replicator(data, shapes):
    for name, shape in shapes.items():
        if name not in data:
            log.info(f"Input '{name}' from shapes was not found in data")
            continue
        err_msg = 'Final batch alignment error for layer `{}`: '.format(name)

        data[name] = np.array(data[name])
        old_shape = np.array(data[name].shape)
        new_shape = np.array(shapes[name])

        if old_shape.size != new_shape.size:
            # Rank resize. We assume that it is Faster-like input with input shape
            if np.prod(old_shape) == np.prod(new_shape):
                data[name].reshape(new_shape)
                old_shape = new_shape

        assert old_shape.size == new_shape.size, 'Rank resize detected'
        if np.all((new_shape % old_shape) == 0):
            assert np.all(old_shape <= new_shape), 'Reshaping to shape that is less than original network shape'
            log.info('New shape is evenly divided by original network shape')
            multiplier = tuple(np.array(new_shape / old_shape, dtype=np.int_))
            data[name] = np.tile(data[name], multiplier)
        else:
            # TF OD models can not be reshaped in 2x bacause they should keep aspect ratio
            log.info('New shape is not evenly divided by original network shape data_shape={}, net_shape={}'
                     ''.format(data[name].shape, new_shape))
            assert len(new_shape) == 4, \
                "Unsupported by tests reshape: Non 4D input {}, original shape {}".format(new_shape, old_shape)

            multiplier = tuple(np.array(new_shape // old_shape + np.ones(new_shape.size), dtype=np.int))
            replicated_data = np.tile(data[name], multiplier)
            data[name] = replicated_data[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

        assert np.array_equal(data[name].shape, new_shape), \
            err_msg + 'data_shape={}, net_shape={}'.format(data[name].shape, new_shape)

    log.info('Input data was aligned with shapes=`{}`, new_data_shapes=`{}`'
             ''.format(shapes, {k: v.shape for k, v in data.items()}))
    return data

