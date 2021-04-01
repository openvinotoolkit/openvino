"""
 Copyright (C) 2018-2021 Intel Corporation

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

import numpy as np

from mo.ops.op import Op


class ExperimentalDetectronROIFeatureExtractor(Op):
    op = 'ExperimentalDetectronROIFeatureExtractor'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            version='opset6',
            infer=__class__.infer,
            out_ports_count=2,
        )

        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            ('pyramid_scales', lambda node: ','.join(map(str, node['pyramid_scales']))),
            'output_size',
            'sampling_ratio',
            ('aligned', lambda node: str(bool(node.soft_get('aligned', False))).lower())]

    @staticmethod
    def infer(node):
        input_rois_shape = node.in_node(0).shape
        rois_num = input_rois_shape[0]
        input_features_level_0_shape = node.in_node(1).shape
        channels_num = input_features_level_0_shape[1]
        node.out_node(0).shape = np.array([rois_num, channels_num, node.output_size, node.output_size], dtype=np.int64)
        if not node.out_port(1).disconnected():
            node.out_node(1).shape = np.array([rois_num, 4], dtype=np.int64)
