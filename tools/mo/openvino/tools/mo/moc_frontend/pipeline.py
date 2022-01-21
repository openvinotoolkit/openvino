# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
from typing import List
import sys
from os import environ

from openvino.tools.mo.moc_frontend.analysis import json_model_analysis_dump
from openvino.tools.mo.moc_frontend.extractor import fe_user_data_repack
from openvino.tools.mo.middle.passes.infer import validate_batch_in_shape
from openvino.tools.mo.utils.class_registration import get_enabled_and_disabled_transforms

from openvino.runtime import Dimension, PartialShape        # pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEnd, InputModel, NotImplementedFailure, Place # pylint: disable=no-name-in-module,import-error
from openvino.runtime.utils.types import get_element_type   # pylint: disable=no-name-in-module,import-error

import numpy as np


def moc_pipeline(argv: argparse.Namespace, moc_front_end: FrontEnd):
    """
    Load input model and convert it to nGraph function
    :param: argv: parsed command line arguments
    :param: moc_front_end: Loaded Frontend for converting input model
    :return: converted nGraph function ready for serialization
    """
    input_model = moc_front_end.load(argv.input_model)

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
            if not hasattr(new_input, 'input_name'):
                continue

            try:
                model.add_name_for_tensor(new_input['node'], new_input['input_name'])
            except NotImplementedFailure as e:
                # some frontends might not implement this method
                log.warn('Could not add an additional name to a tensor pointed to by \'{}\'. Details: {}'.format(
                    new_input['input_name'], str(e)))

    enabled_transforms, disabled_transforms = get_enabled_and_disabled_transforms()
    if 'ANALYSIS_JSON_PRINT' in enabled_transforms:
        # NOTE that model analysis is performed before applying user's settings (inputs's shapes etc.)
        framework_model = moc_front_end.decode(input_model)
        json_model_analysis_dump(framework_model)
        # a model is not processed further in json analysis mode
        sys.exit(0)

    inputs_equal = True
    if user_shapes:
        inputs_equal = check_places_are_same(input_model.get_inputs(), user_shapes)

    outputs_equal = True
    if outputs:
        outputs_equal = check_places_are_same(input_model.get_outputs(), outputs)
    log.debug('Inputs are same: {}, outputs are same: {}'.format(
        inputs_equal, outputs_equal))

    if not inputs_equal and not outputs_equal:
        log.debug('Using extract subgraph')
        new_input_places = [x['node'] for x in user_shapes]
        new_output_places = [x['node'] for x in outputs]
        add_names_to_tensors(input_model, user_shapes)
        input_model.extract_subgraph(new_input_places, new_output_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
                input_model, argv.placeholder_shapes, argv.placeholder_data_types,
                argv.output, argv.freeze_placeholder_with_value, moc_front_end.get_name())
    elif not inputs_equal:
        log.debug('Using override_all_inputs')
        add_names_to_tensors(input_model, user_shapes)
        new_input_places = [x['node'] for x in user_shapes]
        input_model.override_all_inputs(new_input_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
                input_model, argv.placeholder_shapes, argv.placeholder_data_types,
                argv.output, argv.freeze_placeholder_with_value, moc_front_end.get_name())
    elif not outputs_equal:
        log.debug('Using override_all_outputs')
        add_names_to_tensors(input_model, user_shapes)
        new_output_places = [x['node'] for x in outputs]
        input_model.override_all_outputs(new_output_places)

    if user_shapes:
        for user_shape in user_shapes:
            if user_shape.get('shape') is not None:
                input_model.set_partial_shape(
                    user_shape['node'], partial_shape_from_tuple(user_shape['shape']))
            if user_shape.get('data_type') is not None:
                data_type = get_element_type(user_shape['data_type'])
                log.debug('Set data type: {}'.format(data_type))
                input_model.set_element_type(user_shape['node'], data_type)

    def shape_to_array(shape: PartialShape):
        return [shape.get_dimension(i) for i in range(shape.rank.get_length())]

    # Set batch size
    if argv.batch is not None and argv.batch > 0:
        log.debug('Setting batch size to {}'.format(argv.batch))
        for place in input_model.get_inputs():
            old_partial_shape = input_model.get_partial_shape(place)
            old_shape_array = shape_to_array(old_partial_shape) if old_partial_shape.rank.is_static else []
            joined_name = ' '.join(place.get_names())
            validate_batch_in_shape(old_shape_array, joined_name)

            # Assume batch size is always 1-st dimension in shape
            # Keep other dimensions unchanged
            new_shape = [old_partial_shape.get_dimension(i)
                         for i in range(old_partial_shape.rank.get_length())]
            new_shape[0] = Dimension(argv.batch)

            new_partial_shape = PartialShape(new_shape)
            log.debug('Input: {}, Old shape: {}, New shape: {}'.format(
                joined_name, old_shape_array, new_shape))
            input_model.set_partial_shape(place, new_partial_shape)

    ngraph_function = moc_front_end.convert(input_model)
    return ngraph_function


def partial_shape_from_tuple(shape: tuple):
    new_shape = []
    for dim in shape:
        if isinstance(dim, tuple):
            assert len(dim) == 2, "Incorrect boundaries of dimension {} in shape {}".format(dim, shape)
            assert dim[0] >= 0, "Incorrect min value of dimension {} in shape".format(dim, shape)
            new_shape.append(Dimension(dim[0], dim[1]))
        else:
            assert isinstance(dim, np.int64), "Incorrect type of dimension {} in shape".format(dim, shape)
            new_shape.append(Dimension(dim))
    return PartialShape(new_shape)
