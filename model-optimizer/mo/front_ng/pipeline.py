# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import logging as log
from mo.front_ng.extractor import fe_user_data_repack
from mo.middle.passes.infer import validate_batch_in_shape


def moc_pipeline(argv: argparse.Namespace):
    from ngraph import Dimension, PartialShape        # pylint: disable=no-name-in-module,import-error
    from ngraph.utils.types import get_element_type   # pylint: disable=no-name-in-module,import-error
    log.info('New MOC pipeline')
    fem = argv.feManager
    log.info(f'fem.availableFrontEnds: {str(fem.availableFrontEnds())}')
    log.info(f'Initializing new FE for framework {argv.framework}')
    fe = fem.loadByFramework(argv.framework)
    inputModel = fe.loadFromFile(argv.input_model)

    user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
        inputModel, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.output, argv.freeze_placeholder_with_value)

    def compare_nodes(old, new):
        eq = len(old) == len(new)
        if eq:
            for item in old:
                found = [x for x in new if x['node'].isEqual(item)]
                if not found:
                    eq = False
                    break
        return eq

    inputsEqual = True
    if user_shapes:
        inputsEqual = compare_nodes(inputModel.getInputs(), user_shapes)

    outputsEqual = True
    if outputs:
        outputsEqual = compare_nodes(inputModel.getOutputs(), outputs)
    log.debug(f"Inputs are same: {inputsEqual}, outputs are same: {outputsEqual}")

    if not inputsEqual and not outputsEqual:
        # Use ExtractSubgraph
        newInputPlaces = [x['node'] for x in user_shapes]
        newOutputPlaces = [x['node'] for x in outputs]
        log.debug("Using extract subgraph")
        log.debug(f"Inputs: {newInputPlaces}")
        log.debug(f"Outputs: {newOutputPlaces}")
        inputModel.extractSubgraph(newInputPlaces, newOutputPlaces)
    elif not inputsEqual:
        newInputPlaces = [x['node'] for x in user_shapes]
        log.debug("Using overrideAllInputs")
        log.debug(f"Inputs: {newInputPlaces}")
        inputModel.overrideAllInputs(newInputPlaces)
    elif not outputsEqual:
        newOutputPlaces = [x['node'] for x in outputs]
        log.debug("Using overrideAllOutputs")
        log.debug(f"Outputs: {newOutputPlaces}")
        inputModel.overrideAllOutputs(newOutputPlaces)

    if user_shapes:
        for user_shape in user_shapes:
            if 'shape' in user_shape and user_shape['shape'] is not None:
                inputModel.setPartialShape(user_shape['node'], PartialShape(user_shape['shape']))
            if 'data_type' in user_shape and user_shape['data_type'] is not None:
                data_type = get_element_type(user_shape['data_type'])
                log.debug(f"Set data type: {data_type}")
                inputModel.setElementType(user_shape['node'], data_type)

    # Set batch size
    if argv.batch is not None and argv.batch > 0:
        log.debug(f"Setting batch size to {argv.batch}")
        for place in inputModel.getInputs():
            oldPartShape = inputModel.getPartialShape(place)
            newshape = []
            oldshape_converted = []
            joinedName = ' '.join(place.getNames())
            if oldPartShape.rank.is_static:
                for i in range(oldPartShape.rank.get_length()):
                    # Assume batch size is always 1-st dimension in shape
                    # Keep other dimensions unchanged
                    newshape.append(Dimension(argv.batch) if i is 0 else oldPartShape.get_dimension(i))
                    oldshape_converted.append(oldPartShape.get_dimension(i))

                validate_batch_in_shape(oldshape_converted, joinedName)
            else:
                # In case of fully dynamic shape raise the same error as for invalid batch dimension
                validate_batch_in_shape(oldshape_converted, joinedName)

            newPartShape = PartialShape(newshape)
            log.debug(f"Input: {joinedName}, Old shape: {oldshape_converted}, New shape: {newshape}")
            inputModel.setPartialShape(place, newPartShape)

    nGraphFunction = fe.convert(inputModel)
    return nGraphFunction
