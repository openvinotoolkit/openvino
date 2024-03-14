# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.common import tf_data_type_decode


class CastFrontExtractor(FrontExtractorOp):
    op = 'Cast'
    enabled = True

    @classmethod
    def extract(cls, node):
        cast_dst_type = tf_data_type_decode[node.pb.attr['DstT'].type][0]
        Cast.update_node_stat(node, {'dst_type': cast_dst_type})
        return cls.enabled
