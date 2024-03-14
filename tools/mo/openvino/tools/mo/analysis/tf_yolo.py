# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pattern_match import apply_pattern
from openvino.tools.mo.utils.model_analysis import AnalyzeAction, graph_contains_scope


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


YOLO_CONFIGS = {'YOLOV2Full': ['openvino/tools/mo/front/tf/yolo_v2.json', 'openvino/tools/mo/front/tf/yolo_v2_voc.json'],
                'YOLOV3Full': ['openvino/tools/mo/front/tf/yolo_v3.json', 'openvino/tools/mo/front/tf/yolo_v3_voc.json'],
                'YOLOV2Tiny': ['openvino/tools/mo/front/tf/yolo_v2_tiny.json', 'openvino/tools/mo/front/tf/yolo_v2_tiny_voc.json'],
                'YOLOV3Tiny': ['openvino/tools/mo/front/tf/yolo_v3_tiny.json', 'openvino/tools/mo/front/tf/yolo_v3_tiny_voc.json'],
                }


def get_YOLO_params_by_flavor(flavor: str):
    result = dict()
    result['flavor'] = flavor
    result['mandatory_parameters'] = {'transformations_config': YOLO_CONFIGS[flavor]}
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
            message = "Your model looks like YOLOv1 or YOLOv2 Model.\n" \
                      "To generate the IR, provide TensorFlow YOLOv1 or YOLOv2 Model to the Model Optimizer with the following parameters:\n" \
                      "\t--input_model <path_to_model>/<model_name>.pb\n" \
                      "\t--batch 1\n" \
                      "\t--transformations_config <PYTHON_SITE_PACKAGES>/openvino/tools/mo/front/tf/<yolo_config>.json\n" \
                      "All detailed information about conversion of this model can be found at\n" \
                      "https://docs.openvino.ai/2023.0/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html"
            return {'model_type': {'YOLO': get_YOLO_params_by_flavor(flavor)}}, message
        else:
            return None, None


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
            message = "Your model looks like YOLOv3 Model.\n" \
                      "To generate the IR, provide TensorFlow YOLOv3 Model to the Model Optimizer with the following parameters:\n" \
                      "\t--input_model <path_to_model>/yolo_v3.pb\n" \
                      "\t--batch 1\n" \
                      "\t--transformations_config <PYTHON_SITE_PACKAGES>/openvino/tools/mo/front/tf/yolo_v3.json\n" \
                      "Detailed information about conversion of this model can be found at\n" \
                      "https://docs.openvino.ai/2023.0/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html"
            return {'model_type': {'YOLO': get_YOLO_params_by_flavor(flavor)}}, message
        else:
            return None, None
