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
import logging as log

from mo.graph.graph import Graph
from mo.utils.model_analysis import AnalyzeAction, graph_contains_scope
from mo.utils.utils import files_by_pattern, get_mo_root_dir


class TensorFlowObjectDetectionAPIAnalysis(AnalyzeAction):
    """
    The analyser checks if the provided model is TF OD API model from
    https://github.com/tensorflow/models/tree/master/research/object_detection/g3doc/detection_model_zoo.md of one of 4
    supported flavors: SSD, RFCN, Faster RCNN, Mask RCNN.
    """
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']

    model_scopes = [('MaskRCNN', ['Preprocessor',
                                  'FirstStageFeatureExtractor',
                                  'SecondStageFeatureExtractor',
                                  'SecondStageBoxPredictor',
                                  'SecondStageBoxPredictor_1',
                                  'SecondStageFeatureExtractor_1',
                                  ]),
                    ('RFCN', ['Preprocessor',
                              'FirstStageFeatureExtractor',
                              'SecondStageFeatureExtractor',
                              'SecondStageBoxPredictor',
                              'SecondStageBoxPredictor/map',
                              'SecondStageBoxPredictor/map_1',
                              'SecondStagePostprocessor',
                              ]),
                    ('FasterRCNN', ['Preprocessor',
                                    'FirstStageFeatureExtractor',
                                    'SecondStageFeatureExtractor',
                                    'SecondStageBoxPredictor',
                                    'SecondStagePostprocessor',
                                    ]),
                    ('SSD', ['Preprocessor',
                             'FeatureExtractor',
                             'Postprocessor',
                             ]),
                    ]

    file_patterns = {'MaskRCNN': 'mask_rcnn_support.*\\.json',
                     'RFCN': 'rfcn_support.*\\.json',
                     'FasterRCNN': 'faster_rcnn_support.*\\.json',
                     'SSD': 'ssd.*_support.*\\.json',
                     }

    def analyze(self, graph: Graph):
        if any([name not in graph.nodes() for name in ['image_tensor', 'detection_classes', 'detection_boxes',
                                                       'detection_scores']]):
            log.debug('The model does not contain nodes that must exist in the TF OD API models')
            return None

        for flavor, scopes in __class__.model_scopes:
            if all([graph_contains_scope(graph, scope) for scope in scopes]):
                result = dict()
                result['flavor'] = flavor
                result['mandatory_parameters'] = {'tensorflow_use_custom_operations_config':
                                                      files_by_pattern(get_mo_root_dir() + '/extensions/front/tf',
                                                                       __class__.file_patterns[flavor],
                                                                       add_prefix=True),
                                                  'tensorflow_object_detection_api_pipeline_config': None,
                                                  }
                return {'model_type': {'TF_OD_API': result}}
        return None
