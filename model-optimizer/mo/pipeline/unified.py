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
    ie = IECore()
    graph = ie.read_network(model=argv.input_model)
    #TODO: provide real shapes here
    print('Placeholder shapes:' + str(argv.placeholder_shapes))
    #print('Placeholder shapes:' + str(argv.user_shapes))
    graph.reshape({'image': argv.placeholder_shapes})
    return graph