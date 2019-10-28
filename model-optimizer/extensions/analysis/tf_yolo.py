"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph
from mo.middle.pattern_match import apply_pattern
from mo.utils.model_analysis import AnalyzeAction, graph_contains_scope


YOLO_PATTERN = {
    'nodes': [
        ('pad', dict(op='Pad')),
        ('conv', dict(op='Conv2D')),
        ('sub', dict(op='Sub')),
        ('div', dict(op='Div')),
        ('mul', dict(op='Mul')),
        ('bias_add', dict(op='Add')),
        ('mul_2', dict(op='Mul')),
        ('max', dict(op='Maximum')),
    ],
    'edges': [
        ('pad', 'conv', {'out': 0}),
        ('conv', 'sub', {'out': 0}),
        ('sub', 'div', {'out': 0}),
        ('div', 'mul', {'out': 0}),
        ('mul', 'bias_add', {'out': 0}),
        ('bias_add', 'mul_2', {'out': 0}),
        ('bias_add', 'max', {'out': 0}),
        ('mul_2', 'max', {'out': 0}),
    ]
}


def pattern_instance_counter(graph: Graph, match: dict):
    pattern_instance_counter.counter += 1


pattern_instance_counter.counter = 0


YOLO_CONFIGS = {'YOLOV2Full': ['extensions/front/tf/yolo_v2.json', 'extensions/front/tf/yolo_v2_voc.json'],
                'YOLOV3Full': ['extensions/front/tf/yolo_v3.json', 'extensions/front/tf/yolo_v3_voc.json'],
                'YOLOV2Tiny': ['extensions/front/tf/yolo_v2_tiny.json', 'extensions/front/tf/yolo_v2_tiny_voc.json'],
                'YOLOV3Tiny': ['extensions/front/tf/yolo_v3_tiny.json', 'extensions/front/tf/yolo_v3_tiny_voc.json'],
                }


def get_YOLO_params_by_flavor(flavor: str):
    result = dict()
    result['flavor'] = flavor
    result['mandatory_parameters'] = {'tensorflow_use_custom_operations_config': YOLO_CONFIGS[flavor]}
    return result


class TensorFlowYOLOV1V2Analysis(AnalyzeAction):
    """
    The analyser checks if the provided model is TensorFlow YOLO models from https://github.com/thtrieu/darkflow .
    """
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def analyze(self, graph: Graph):
        pattern_instance_counter.counter = 0
        apply_pattern(graph, **YOLO_PATTERN, action=pattern_instance_counter)

        flavor = None
        if pattern_instance_counter.counter > 0:
            if pattern_instance_counter.counter == 22:
                flavor = 'YOLOV2Full'
            elif pattern_instance_counter.counter == 8:
                flavor = 'YOLOV2Tiny'

        if flavor is not None:
            return {'model_type': {'YOLO': get_YOLO_params_by_flavor(flavor)}}
        else:
            return None


class TensorFlowYOLOV3Analysis(AnalyzeAction):
    """
    The analyser checks if the provided model is TensorFlow YOLO models from
    https://github.com/mystic123/tensorflow-yolo-v3.
    """
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    def analyze(self, graph: Graph):
        flavor = None
        if graph_contains_scope(graph, 'detector/yolo-v3') and graph_contains_scope(graph, 'detector/darknet-53'):
            flavor = 'YOLOV3Full'
        elif graph_contains_scope(graph, 'detector/yolo-v3-tiny'):
            flavor = 'YOLOV3Tiny'

        if flavor is not None:
            return {'model_type': {'YOLO': get_YOLO_params_by_flavor(flavor)}}
        else:
            return None
