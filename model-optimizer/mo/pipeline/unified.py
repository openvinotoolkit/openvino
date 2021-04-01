# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from mo.graph.graph import Graph
from mo.pipeline.common import get_ir_version
from mo.utils import class_registration

from extensions.front.user_data_repack import UserDataRepack
from extensions.front.input_cut import InputCut
from extensions.front.output_cut import OutputCut
from mo.utils.class_registration import apply_replacements_list
from ngraph import FrontEndManager
from ngraph import function_to_cnn
from ngraph import PartialShape
import logging as log


def unified_pipeline(argv: argparse.Namespace):
    log.info('Legacy unified pipeline')
    graph = Graph(cmd_params=argv, name=argv.model_name, ir_version=get_ir_version(argv))
    class_registration.apply_replacements(graph, [
        class_registration.ClassType.LOADER,
        class_registration.ClassType.FRONT_REPLACER,
        class_registration.ClassType.MIDDLE_REPLACER,
        class_registration.ClassType.BACK_REPLACER
    ])
    return graph

def moc_pipeline(argv: argparse.Namespace):
    log.info('New MOC pipeline')
    fem = argv.feManager if 'feManager' in argv else FrontEndManager()
    log.info('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    log.info('Initializing new FE for framework {}'.format(argv.framework))
    fe = fem.loadByFramework(argv.framework)
    print(fe)
    inputModel = fe.loadFromFile(argv.input_model)


    # Wrap nGraph network to Graph for smoothly pass through the legacy code in MO.
    # This trick doesn't mean that we will hold Graph forever as a wrapper, it is derived from
    # NX graph and this is not required. But Graph has several methods that can be implemented for nGraph
    # and probably they should be kept at least for transition period where some existing transformations
    # that manipulate Graph object really translate those modifications directly to nGraph representation.
    graph = Graph(frontend=fe, input_model=inputModel, cmd_params=argv, name=argv.model_name, ir_version=get_ir_version(argv))

    transforms = [
        UserDataRepack,
        #InputCut,
        #OutputCut
    ]

    apply_replacements_list(graph, transforms)
    user_shapes = graph.graph['user_shapes']
    if len(user_shapes) > 0:
        assert len(inputModel.getInputs()) == 1
        assert len(user_shapes) == 1
        inputModel.setPartialShape(user_shapes[0]['node'], PartialShape(user_shapes[0]['shape']))
    nGraphModel = fe.convert(inputModel)
    network = function_to_cnn(nGraphModel)
    graph.graph['network'] = network
    return graph