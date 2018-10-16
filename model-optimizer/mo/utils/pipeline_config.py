"""
 Copyright (c) 2018 Intel Corporation

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

from mo.utils.error import Error
from mo.utils.simple_proto_parser import SimpleProtoParser


class PipelineConfig:
    """
    The class that parses pipeline.config files used to generate TF models generated using Object Detection API.
    The class stores data read from the file in a plain dictionary for easier access using the get_param function.
    """
    _raw_data_dict = dict()
    _model_params = dict()

    def __init__(self, file_name: str):
        self._raw_data_dict = SimpleProtoParser().parse_file(file_name)
        if not self._raw_data_dict:
            raise Error('Failed to parse pipeline.config file {}'.format(file_name))

        self._initialize_model_params()

    def _initialize_model_params(self):
        """
        Store global params in the dedicated dictionary self._model_params for easier use.
        :return: None
        """
        params = list(self._raw_data_dict['model'].values())[0]

        # global topology parameters
        self._model_params['num_classes'] = params['num_classes']

        # pre-processing of the image
        self._model_params['image_resizer'] = list(params['image_resizer'].keys())[0]
        image_resize_params = list(params['image_resizer'].values())[0]
        if self._model_params['image_resizer'] == 'keep_aspect_ratio_resizer':
            self._model_params['preprocessed_image_height'] = image_resize_params['min_dimension']
            self._model_params['preprocessed_image_width'] = self._model_params['preprocessed_image_height']
        elif self._model_params['image_resizer'] == 'fixed_shape_resizer':
            self._model_params['preprocessed_image_height'] = image_resize_params['height']
            self._model_params['preprocessed_image_width'] = image_resize_params['width']
        else:
            raise Error('Unknown image resizer type "{}"'.format(self._model_params['image_resizer']))

        # grid anchors generator
        if 'first_stage_anchor_generator' in params:
            grid_params = params['first_stage_anchor_generator']['grid_anchor_generator']
            self._model_params['anchor_generator_base_size'] = 256
            self._model_params['anchor_generator_stride'] = grid_params['height_stride']
            self._model_params['anchor_generator_scales'] = grid_params['scales']
            self._model_params['anchor_generator_aspect_ratios'] = grid_params['aspect_ratios']

        if 'feature_extractor' in params:
            if 'first_stage_features_stride' in params['feature_extractor']:
                self._model_params['features_extractor_stride'] = params['feature_extractor']['first_stage_features_stride']
            else:  # the value is not specified in the configuration file for NASNet so use default value here
                self._model_params['features_extractor_stride'] = 16

        # Proposal and ROI Pooling layers
        for param in ['first_stage_nms_score_threshold', 'first_stage_nms_iou_threshold', 'first_stage_max_proposals',
                      'initial_crop_size']:
            if param in params:
                self._model_params[param] = params[param]

        # post-processing parameters
        postprocessing_params = None
        for postpocessing_type in ['post_processing', 'second_stage_post_processing']:
            if postpocessing_type in params:
                postprocessing_params = params[postpocessing_type]['batch_non_max_suppression']
                self._model_params['postprocessing_score_converter'] = params[postpocessing_type]['score_converter']
        if postprocessing_params is not None:
            for param in ['score_threshold', 'iou_threshold', 'max_detections_per_class', 'max_total_detections']:
                self._model_params['postprocessing_' + param] = postprocessing_params[param]

    def get_param(self, param: str):
        if param not in self._model_params:
            return None
        return self._model_params[param]
