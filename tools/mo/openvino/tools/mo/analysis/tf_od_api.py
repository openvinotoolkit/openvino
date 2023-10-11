# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.model_analysis import AnalyzeAction, graph_contains_scope
from openvino.tools.mo.utils.utils import files_by_pattern, get_mo_root_dir


class TensorFlowObjectDetectionAPIAnalysis(AnalyzeAction):
    """
    The analyser checks if the provided model is TF OD API model from
    https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc/detection_model_zoo.md of one of 4
    supported flavors: SSD, RFCN, Faster RCNN, Mask RCNN.
    """
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    file_patterns = {'MaskRCNN': 'mask_rcnn_support.*\\.json',
                     'RFCN': 'rfcn_support.*\\.json',
                     'FasterRCNN': 'faster_rcnn_support.*\\.json',
                     'SSD': 'ssd.*_support.*\\.json',
                     }
    model_scopes = {'MaskRCNN': (['FirstStageFeatureExtractor',
                                  'SecondStageFeatureExtractor',
                                  'SecondStageBoxPredictor',
                                  'SecondStageBoxPredictor_1',
                                  'SecondStageFeatureExtractor_1',
                                  ],),
                    'RFCN': (['FirstStageFeatureExtractor',
                              'SecondStageFeatureExtractor',
                              'SecondStageBoxPredictor',
                              'SecondStageBoxPredictor/map',
                              'SecondStageBoxPredictor/map_1',
                              'SecondStagePostprocessor',
                              ],),
                    'FasterRCNN': (['FirstStageFeatureExtractor',
                                    'SecondStageFeatureExtractor',
                                    'SecondStageBoxPredictor',
                                    'SecondStagePostprocessor',
                                    ],
                                   ['FirstStageRPNFeatures',
                                    'FirstStageBoxPredictor',
                                    'SecondStagePostprocessor',
                                    'mask_rcnn_keras_box_predictor',
                                    ],),
                    'SSD': ([('FeatureExtractor', 'ssd_mobile_net_v2keras_feature_extractor',
                              'ssd_mobile_net_v1fpn_keras_feature_extractor',
                              'ssd_mobile_net_v2fpn_keras_feature_extractor', 'ResNet50V1_FPN', 'ResNet101V1_FPN',
                              'ResNet152V1_FPN'
                              ),
                             'Postprocessor']
                            ),
                    }

    def analyze(self, graph: Graph):
        tf_1_names = ['image_tensor', 'detection_classes', 'detection_boxes', 'detection_scores',
                      ('Preprocessor', 'map')]
        tf_1_cond = all([graph_contains_scope(graph, scope) for scope in tf_1_names])

        tf_2_names = ['input_tensor', 'output_control_node', 'Identity_', ('Preprocessor', 'map')]
        tf_2_cond = all([graph_contains_scope(graph, scope) for scope in tf_2_names])

        if not tf_1_cond and not tf_2_cond:
            log.debug('The model does not contain nodes that must exist in the TF OD API models')
            return None, None

        for flavor, scopes_tuple in self.model_scopes.items():
            for scopes in scopes_tuple:
                if all([graph_contains_scope(graph, scope) for scope in scopes]):
                    result = dict()
                    result['flavor'] = flavor
                    result['mandatory_parameters'] = {'transformations_config':
                                                          files_by_pattern(get_mo_root_dir() + '/openvino/tools/mo/front/tf',
                                                                           __class__.file_patterns[flavor],
                                                                           add_prefix=True),
                                                      'tensorflow_object_detection_api_pipeline_config': None,
                                                      }
                    message = "Your model looks like TensorFlow Object Detection API Model.\n" \
                              "Check if all parameters are specified:\n" \
                              "\t--transformations_config\n" \
                              "\t--tensorflow_object_detection_api_pipeline_config\n" \
                              "\t--input_shape (optional)\n" \
                              "\t--reverse_input_channels (if you convert a model to use with the Inference Engine sample applications)\n" \
                              "Detailed information about conversion of this model can be found at\n" \
                              "https://docs.openvino.ai/2023.0/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html"
                    return {'model_type': {'TF_OD_API': result}}, message
        return None, None
