
"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.psroipooling import DeformablePSROIPoolingOp, PSROIPoolingOp
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class DeformablePSROIPoolingFrontExtractor(FrontExtractorOp):
    op = '_contrib_DeformablePSROIPooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        spatial_scale = attrs.float('spatial_scale', None)
        group_size = attrs.int('group_size', 0)
        no_trans = attrs.bool('no_trans', False)
        trans_std = attrs.float('trans_std', 0)
        output_dim = attrs.int('output_dim', 0)
        part_size = attrs.int('part_size', 0)
        sample_per_part = attrs.int('sample_per_part', 1)
        pooled_size = attrs.int('pooled_size', 0)

        data = {
            'spatial_scale': spatial_scale,
            'mode': 'bilinear_deformable',
            'group_size': group_size,
            'output_dim': output_dim,
            'trans_std': trans_std,
            'part_size': part_size,
            'spatial_bins_x': sample_per_part,
            'spatial_bins_y': sample_per_part,
            'pooled_width': pooled_size,
            'pooled_height': pooled_size,
        }

        # update the attributes of the node
        if not node.graph.graph['cmd_params'].generate_experimental_IR_V10:
            data.update({'no_trans': no_trans})
            PSROIPoolingOp.update_node_stat(node, data)
        else:
            DeformablePSROIPoolingOp.update_node_stat(node, data)
        return cls.enabled
