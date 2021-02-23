"""
 Copyright (C) 2020 Intel Corporation

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

from mo.graph.graph import Graph
from mo.pipeline.common import get_ir_version
from mo.utils import class_registration

from extensions.front.user_data_repack import UserDataRepack
from extensions.front.input_cut import InputCut
from extensions.front.output_cut import OutputCut
from mo.utils.class_registration import apply_replacements_list


def unified_pipeline(argv: argparse.Namespace):
    graph = Graph(cmd_params=argv, name=argv.model_name, ir_version=get_ir_version(argv))
    class_registration.apply_replacements(graph, [
        class_registration.ClassType.LOADER,
        class_registration.ClassType.FRONT_REPLACER,
        class_registration.ClassType.MIDDLE_REPLACER,
        class_registration.ClassType.BACK_REPLACER
    ])
    return graph

def moc_pipeline(argv: argparse.Namespace):
    from openvino.inference_engine import IECore
    from openvino.inference_engine import IENetwork
    import ngraph as ng
    print('ngrph.dir = ' + str(dir(ng)))
    from ngraph import FrontEndManager
    ie = IECore()
    fem = FrontEndManager()
    print('fem.availableFrontEnds: ' + str(fem.availableFrontEnds()))
    fe = fem.loadByFramework('onnx')
    print(fe)
    inputModel = fe.load(argv.input_model)
    nGraphModel = fe.convert(inputModel)
    #network = ng.function_to_cnn(nGraphModel)
    #network = ie.read_network(model=argv.input_model)

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
    print(user_shapes)
    if user_shapes:
        inputModel.extractSubgraph([us['node'] for us in user_shapes], [])
    nGraphModel = fe.convert(inputModel)
    network = ng.function_to_cnn(nGraphModel)
    graph.graph['network'] = network
    return graph