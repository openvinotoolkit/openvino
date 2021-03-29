# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.spatial_transformer import SpatialTransformOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.extractor import FrontExtractorOp


class SpatialTransformFrontExtractor(FrontExtractorOp):
    op = 'SpatialTransformer'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.st_param

        update_attrs = {
            'transform_type': param.transform_type,
            'sampler_type': param.sampler_type,
            'output_H': param.output_H,
            'output_W': param.output_W,
            'to_compute_dU': int(param.to_compute_dU),
            'theta_1_1': param.theta_1_1,
            'theta_1_2': param.theta_1_2,
            'theta_1_3': param.theta_1_3,
            'theta_2_1': param.theta_2_1,
            'theta_2_2': param.theta_2_2,
            'theta_2_3': param.theta_2_3
        }

        mapping_rule = merge_attrs(param, update_attrs)

        # update the attributes of the node
        SpatialTransformOp.update_node_stat(node, mapping_rule)
        return cls.enabled
