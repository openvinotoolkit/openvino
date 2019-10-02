"""
 Copyright (c) 2018-2019 Intel Corporation

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
import re

from mo.utils.error import Error
from mo.utils.simple_proto_parser import SimpleProtoParser


# The list of rules how to map the value from the pipeline.config file to the dictionary with attributes.
# The rule is either a string or a tuple with two elements. In the first case the rule string is used as a key to
# search in the parsed pipeline.config file attributes dictionary and a key to save found value. In the second case the
# first element of the tuple is the key to save found value; the second element of the tuple is a string defining the
# path to the value of the attribute in the pipeline.config file. The path consists of the regular expression strings
# defining the dictionary key to look for separated with a '/' character.
mapping_rules = [
    'num_classes',
    # preprocessing block attributes
    ('resizer_image_height', 'image_resizer/fixed_shape_resizer/height'),
    ('resizer_image_width', 'image_resizer/fixed_shape_resizer/width'),
    ('resizer_min_dimension', 'image_resizer/keep_aspect_ratio_resizer/min_dimension'),
    ('resizer_max_dimension', 'image_resizer/keep_aspect_ratio_resizer/max_dimension'),
    # anchor generator attributes
    ('anchor_generator_height', 'first_stage_anchor_generator/grid_anchor_generator/height$', 256),
    ('anchor_generator_width', 'first_stage_anchor_generator/grid_anchor_generator/width$', 256),
    ('anchor_generator_height_stride', 'first_stage_anchor_generator/grid_anchor_generator/height_stride', 16),
    ('anchor_generator_width_stride', 'first_stage_anchor_generator/grid_anchor_generator/width_stride', 16),
    ('anchor_generator_scales', 'first_stage_anchor_generator/grid_anchor_generator/scales'),
    ('anchor_generator_aspect_ratios', 'first_stage_anchor_generator/grid_anchor_generator/aspect_ratios'),
    ('multiscale_anchor_generator_min_level', 'anchor_generator/multiscale_anchor_generator/min_level'),
    ('multiscale_anchor_generator_max_level', 'anchor_generator/multiscale_anchor_generator/max_level'),
    ('multiscale_anchor_generator_anchor_scale', 'anchor_generator/multiscale_anchor_generator/anchor_scale'),
    ('multiscale_anchor_generator_aspect_ratios', 'anchor_generator/multiscale_anchor_generator/aspect_ratios'),
    ('multiscale_anchor_generator_scales_per_octave', 'anchor_generator/multiscale_anchor_generator/scales_per_octave'),
    # SSD anchor generator attributes
    ('ssd_anchor_generator_min_scale', 'anchor_generator/ssd_anchor_generator/min_scale', 0.2),
    ('ssd_anchor_generator_max_scale', 'anchor_generator/ssd_anchor_generator/max_scale', 0.95),
    ('ssd_anchor_generator_num_layers', 'anchor_generator/ssd_anchor_generator/num_layers'),
    ('ssd_anchor_generator_aspect_ratios', 'anchor_generator/ssd_anchor_generator/aspect_ratios'),
    ('ssd_anchor_generator_scales', 'anchor_generator/ssd_anchor_generator/scales'),
    ('ssd_anchor_generator_interpolated_scale_aspect_ratio',
     'anchor_generator/ssd_anchor_generator/interpolated_scale_aspect_ratio', 1.0),
    ('ssd_anchor_generator_reduce_lowest', 'anchor_generator/ssd_anchor_generator/reduce_boxes_in_lowest_layer'),
    ('ssd_anchor_generator_base_anchor_height', 'anchor_generator/ssd_anchor_generator/base_anchor_height', 1.0),
    ('ssd_anchor_generator_base_anchor_width', 'anchor_generator/ssd_anchor_generator/base_anchor_width', 1.0),
    # Proposal and ROI Pooling layers attributes
    ('first_stage_nms_score_threshold', '.*_nms_score_threshold'),
    ('first_stage_nms_iou_threshold', '.*_nms_iou_threshold'),
    ('first_stage_max_proposals', '.*_max_proposals'),
    ('num_spatial_bins_height', '.*/rfcn_box_predictor/num_spatial_bins_height'),
    ('num_spatial_bins_width', '.*/rfcn_box_predictor/num_spatial_bins_width'),
    ('crop_height', '.*/rfcn_box_predictor/crop_height'),
    ('crop_width', '.*/rfcn_box_predictor/crop_width'),
    'initial_crop_size',
    # Detection Output layer attributes
    ('postprocessing_score_converter', '.*/score_converter'),
    ('postprocessing_score_threshold', '.*/batch_non_max_suppression/score_threshold'),
    ('postprocessing_iou_threshold', '.*/batch_non_max_suppression/iou_threshold'),
    ('postprocessing_max_detections_per_class', '.*/batch_non_max_suppression/max_detections_per_class'),
    ('postprocessing_max_total_detections', '.*/batch_non_max_suppression/max_total_detections'),
    # Variances for predicted bounding box deltas (tx, ty, tw, th)
    ('frcnn_variance_x', 'box_coder/faster_rcnn_box_coder/x_scale', 10.0),
    ('frcnn_variance_y', 'box_coder/faster_rcnn_box_coder/y_scale', 10.0),
    ('frcnn_variance_width', 'box_coder/faster_rcnn_box_coder/width_scale', 5.0),
    ('frcnn_variance_height', 'box_coder/faster_rcnn_box_coder/height_scale', 5.0)
]


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

    @staticmethod
    def _get_value_by_path(params: dict, path: list):
        if not path or len(path) == 0:
            return None
        if not isinstance(params, dict):
            return None
        compiled_regexp = re.compile(path[0])
        for key in params.keys():
            if re.match(compiled_regexp, key):
                if len(path) == 1:
                    return params[key]
                else:
                    value = __class__._get_value_by_path(params[key], path[1:])
                    if value is not None:
                        return value
        return None

    def _update_param_using_rule(self, params: dict, rule: [str, tuple]):
        if isinstance(rule, str):
            if rule in params:
                self._model_params[rule] = params[rule]
                log.debug('Found value "{}" for path "{}"'.format(params[rule], rule))
        elif isinstance(rule, tuple):
            if len(rule) != 2 and len(rule) != 3:
                raise Error('Invalid rule length. Rule must be a tuple with two elements: key and path, or three '
                            'elements: key, path, default_value.')
            value = __class__._get_value_by_path(params, rule[1].split('/'))
            if value is not None:
                log.debug('Found value "{}" for path "{}"'.format(value, rule[1]))
                self._model_params[rule[0]] = value
            elif len(rule) == 3:
                self._model_params[rule[0]] = rule[2]
                log.debug('There is no value path "{}". Set default value "{}"'.format(value, rule[2]))

        else:
            raise Error('Invalid rule type. Rule can be either string or tuple')

    def _initialize_model_params(self):
        """
        Store global params in the dedicated dictionary self._model_params for easier use.
        :return: None
        """

        if 'model' not in self._raw_data_dict:
            raise Error('The "model" key is not found in the configuration file. Looks like the parsed file is not '
                        'Object Detection API model configuration file.')
        params = list(self._raw_data_dict['model'].values())[0]
        for rule in mapping_rules:
            self._update_param_using_rule(params, rule)

    def get_param(self, param: str):
        if param not in self._model_params:
            return None
        return self._model_params[param]
