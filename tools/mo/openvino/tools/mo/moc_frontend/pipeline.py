# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import sys
from copy import copy
from typing import List

import numpy as np
import os

from openvino.frontend import FrontEnd, InputModel, NotImplementedFailure, \
    Place  # pylint: disable=no-name-in-module,import-error
from openvino.runtime import PartialShape, Type  # pylint: disable=no-name-in-module,import-error
from openvino.runtime.utils.types import get_element_type, \
    get_numpy_ctype  # pylint: disable=no-name-in-module,import-error
from openvino.tools.mo.moc_frontend.analysis import json_model_analysis_dump
from openvino.tools.mo.moc_frontend.extractor import fe_user_data_repack, convert_params_lists_to_dicts, fe_output_user_data_repack
from openvino.tools.mo.moc_frontend.layout_utils import update_layout_to_dict, get_dimension_index_by_label
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.type_utils import np_map_cast
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.middle.passes.infer import validate_batch_in_shape


def get_enabled_and_disabled_transforms():
    """
    :return: tuple of lists with force enabled and disabled id of transformations.
    """
    disabled_transforms = os.environ['MO_DISABLED_TRANSFORMS'] if 'MO_DISABLED_TRANSFORMS' in os.environ else ''
    enabled_transforms = os.environ['MO_ENABLED_TRANSFORMS'] if 'MO_ENABLED_TRANSFORMS' in os.environ else ''

    assert isinstance(enabled_transforms, str)
    assert isinstance(disabled_transforms, str)

    disabled_transforms = disabled_transforms.split(',')
    enabled_transforms = enabled_transforms.split(',')

    return enabled_transforms, disabled_transforms


def moc_pipeline(argv: argparse.Namespace, moc_front_end: FrontEnd):
    """
    Load input model and convert it to nGraph function
    :param: argv: parsed command line arguments
    :param: moc_front_end: Loaded Frontend for converting input model
    :return: converted nGraph function ready for serialization
    """
    input_checkpoint = getattr(argv, 'input_checkpoint', None)
    share_weights = getattr(argv, 'share_weights', True)
    if argv.input_model and input_checkpoint:
        # frozen format with v1 checkpoints
        input_model = moc_front_end.load([argv.input_model, argv.input_checkpoint], share_weights)
    elif argv.input_model:
        input_model = moc_front_end.load(argv.input_model, share_weights)
    elif argv.saved_model_dir:
        if argv.saved_model_tags:
            input_model = moc_front_end.load([argv.saved_model_dir, argv.saved_model_tags], share_weights)
        else:
            input_model = moc_front_end.load(argv.saved_model_dir, share_weights)
    elif argv.input_meta_graph:
        input_model = moc_front_end.load(argv.input_meta_graph, share_weights)
        if argv.output:
            # Simulate original behavior with freezing model
            # While freezing we do a cutting of model, to keep similar behavior we
            # need to simulate similar behavior with natively supported model
            outputs = fe_output_user_data_repack(input_model, argv.output, moc_front_end.get_name())
            input_model.override_all_outputs([x['node'] for x in outputs])

    argv.placeholder_shapes, argv.placeholder_data_types, argv.freeze_placeholder_with_value = convert_params_lists_to_dicts(
        input_model, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.freeze_placeholder_with_value, argv.unnamed_freeze_placeholder_with_value)

    user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
        input_model, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.output, argv.freeze_placeholder_with_value, moc_front_end.get_name())

    def check_places_are_same(places_original: List[Place], places_new: List[Place]):
        """
        Check if set of new places is same as original or not.
        :param places_original: List[Place] Original model places
        :param places_new: List[Place] New list of places
        :return: True if new list of places is same as original
        """
        return len(places_original) == len(places_new) and len(
            [item for item in places_original if any(
                [item.is_equal(item2['node']) for item2 in places_new])]) == len(places_original)

    def add_names_to_tensors(model: InputModel, places: List[Place]):
        """
        Adds additional names to some model input tensors. This helper should be used
        when a model modification is going to happen.
        :param model The input model loaded by a given frontend
        :param places An object containing Places and names that will be used for model modification
        """
        for new_input in places:
            if 'input_name' not in new_input:
                continue
            try:
                model.add_name_for_tensor(new_input['node'], new_input['input_name'])
            except NotImplementedFailure as e:
                # some frontends might not implement this method
                log.warning('Could not add an additional name to a tensor pointed to by \'{}\'. Details: {}'.format(
                    new_input['input_name'], str(e)))

    enabled_transforms, disabled_transforms = get_enabled_and_disabled_transforms()
    if 'ANALYSIS_JSON_PRINT' in enabled_transforms:
        # NOTE that model analysis is performed before applying user's settings (inputs's shapes etc.)
        framework_model = moc_front_end.decode(input_model)
        json_model_analysis_dump(framework_model)
        # a model is not processed further in json analysis mode
        sys.exit(0)

    model_inputs = input_model.get_inputs()
    inputs_equal = True
    if user_shapes:
        inputs_equal = check_places_are_same(model_inputs, user_shapes)

    outputs_equal = True
    if outputs:
        outputs_equal = check_places_are_same(input_model.get_outputs(), outputs)
    log.debug('Inputs are same: {}, outputs are same: {}'.format(
        inputs_equal, outputs_equal))

    def create_target_input_shapes(new_input_places):
        if isinstance(new_input_places, list) and len(new_input_places) > 1 \
                and isinstance(new_input_places[0], tuple):
            return new_input_places
        new_input_place_names = [x.get_names()[0] for x in new_input_places]
        shapes = [shape for shape in argv.placeholder_shapes.values()]
        return dict(zip(new_input_place_names, shapes))

    if not inputs_equal and not outputs_equal:
        log.debug('Using extract subgraph')
        new_input_places = [x['node'] for x in user_shapes]
        new_output_places = [x['node'] for x in outputs]
        add_names_to_tensors(input_model, user_shapes)
        input_model.extract_subgraph(new_input_places, new_output_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            placeholder_shapes = create_target_input_shapes(new_input_places)
            new_output_places_name = [x.get_names()[0] for x in new_output_places]

            user_shapes, outputs, _ = fe_user_data_repack(
                input_model, placeholder_shapes, argv.placeholder_data_types,
                new_output_places_name, argv.freeze_placeholder_with_value, moc_front_end.get_name())
    elif not inputs_equal:
        log.debug('Using override_all_inputs')
        add_names_to_tensors(input_model, user_shapes)
        new_input_places = [x['node'] for x in user_shapes]
        input_model.override_all_inputs(new_input_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            placeholder_shapes = create_target_input_shapes(new_input_places)

            user_shapes, outputs, _ = fe_user_data_repack(
                input_model, placeholder_shapes, argv.placeholder_data_types,
                argv.output, argv.freeze_placeholder_with_value, moc_front_end.get_name())
    elif not outputs_equal:
        log.debug('Using override_all_outputs')
        add_names_to_tensors(input_model, user_shapes)
        new_output_places = [x['node'] for x in outputs]
        input_model.override_all_outputs(new_output_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            model_inputs = input_model.get_inputs()

    if user_shapes:
        for user_shape in user_shapes:
            if user_shape.get('shape') is not None:
                input_model.set_partial_shape(
                    user_shape['node'], user_shape['shape'])
            if user_shape.get('data_type') is not None:
                data_type = get_element_type(user_shape['data_type'])
                log.debug('Set data type: {}'.format(data_type))
                input_model.set_element_type(user_shape['node'], data_type)

    if freeze_placeholder:
        for name, value in freeze_placeholder.items():
            node = None
            # look for the certain place in user_shapes
            for node_cur in user_shapes:
                if node_cur.get('input_name') == name or node_cur.get('input_name') == name + ":0":
                    node = node_cur
                    break
            if node is None:
                raise Error("Please check correctness of the command-line. "
                            "Place (operation or tensor) with name {} is not found.".format(name))
            place = node.get('node')

            if node.get('data_type'):
                dtype = node['data_type']
                ov_type = Type(dtype)
            else:
                # we need to detect type of Placeholder
                try:
                    ov_type = input_model.get_element_type(place)
                except NotImplementedFailure:
                    raise Error("Please specify type for value freezing {} node explicitly "
                                "because the frontend does not support automatic type detection.".format(name))
                # in case of cutting graph (or using custom inputs) and unspecified or dynamic type,
                # the default type is fp32
                if ov_type == Type.undefined or ov_type == Type.dynamic:
                    ov_type = Type.f32
                dtype = get_numpy_ctype(ov_type)

            input_model.set_element_type(place, ov_type)
            # prepare and cast value to dtype
            if isinstance(value, list):
                casted_list = list()
                for v in mo_array(value):
                    casted_list.append(np_map_cast[dtype](v))
                value = mo_array(casted_list, dtype=dtype)
            else:
                value = np_map_cast[dtype](value)
            value = np.array(value, dtype=dtype)

            ov_shape = input_model.get_partial_shape(place)
            if node.get('shape'):
                # set user defined shape
                ov_shape = PartialShape(node['shape'])
                input_model.set_partial_shape(place, ov_shape)
            elif ov_shape.is_dynamic:
                # in case of dynamic shape (dynamic rank or dynamic dimension)
                # deduce it based on the value shape and set it
                ov_shape = PartialShape(value.shape)
                input_model.set_partial_shape(place, ov_shape)

            input_model.set_tensor_value(place, value)

    def shape_to_array(shape: PartialShape):
        return [shape.get_dimension(i) for i in range(shape.rank.get_length())]

    # obtain layout for all inputs
    layout_values = {}
    if 'layout_values' in argv and argv.layout_values:
        layout_values = update_layout_to_dict(model_inputs, argv.layout_values,
                                              lambda input_place: input_place.get_names())

    deferred_batch_names = []
    # set batch size for inputs with a static rank
    # for all other inputs, set it after shape deduction is performed during model conversion
    if argv.batch is not None and argv.batch > 0:
        log.debug('Setting batch size to {}'.format(argv.batch))
        frozen_input_names = list(freeze_placeholder.keys()) if freeze_placeholder else []
        for place in model_inputs:
            input_partial_shape = input_model.get_partial_shape(place)
            input_names = place.get_names()
            joined_name = ' '.join(place.get_names())
            assert len(input_names) > 0, "One input place has no names"

            # if this input is frozen, there is no need to set the batch
            is_frozen_input = len([name for name in input_names if name in frozen_input_names]) > 0
            if is_frozen_input:
                # skip the frozen input
                continue

            if not input_partial_shape.rank.is_static:
                # found input with dynamic rank, so have to repeat the batch setting after the model conversion
                deferred_batch_names += input_names
                continue

            batch_dim, is_default_index = get_dimension_index_by_label(input_partial_shape,
                                                                       place.get_names(), layout_values, 'N', 0)
            if batch_dim is None:
                # skip because no batch dimension exists in the input
                continue

            if is_default_index:
                # if the batch index is chosen by default, we need to ensure that its size equals -1, 0 or 1
                validate_batch_in_shape(shape_to_array(input_partial_shape), joined_name)

            assert batch_dim < input_partial_shape.rank.get_length(), \
                "Incorrect layout is specified for {}:" \
                " index of the batch dimension is out of range.".format(input_names[0])

            new_partial_shape = copy(input_partial_shape)
            new_partial_shape[batch_dim] = argv.batch

            log.debug('Input: {}, Old shape: {}, New shape: {}'.format(
                joined_name, input_partial_shape, new_partial_shape))
            input_model.set_partial_shape(place, new_partial_shape)

    ov_model = moc_front_end.convert(input_model)

    if argv.batch is not None and argv.batch > 0 and len(deferred_batch_names) > 0:
        # Frontend convert method can include reverse infer functionality that can deduce undefined input shapes
        # so try to repeat batch setting again
        reshape_dict = {}
        log.debug('Deferred batch setting to size {}'.format(argv.batch))
        is_batch_clarified = False
        for model_input in ov_model.inputs:
            input_name = model_input.any_name
            input_partial_shape = model_input.get_partial_shape()
            if input_name in deferred_batch_names and input_partial_shape.rank.is_static:
                # update input shape with the specified batch for input that originally has dynamic rank
                batch_dim, is_default_index = get_dimension_index_by_label(input_partial_shape,
                                                                           model_input.get_names(),
                                                                           layout_values, 'N', 0)
                if batch_dim is None:
                    continue

                if is_default_index:
                    # if the batch index is chosen by default, we need to ensure that its size equals -1, 0 or 1
                    validate_batch_in_shape(shape_to_array(input_partial_shape), input_name)

                assert batch_dim < input_partial_shape.rank.get_length(), \
                    "Incorrect layout is specified for {}: " \
                    "index of the batch dimension is out of range.".format(input_name)
                input_partial_shape[batch_dim] = argv.batch
                is_batch_clarified = True

            reshape_dict.update({input_name: input_partial_shape})

        if is_batch_clarified:
            # call reshape only if batch dimension for one of the input is clarified
            ov_model.reshape(reshape_dict)

    return ov_model
