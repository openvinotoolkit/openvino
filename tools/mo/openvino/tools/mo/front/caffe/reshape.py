# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.reshape import Reshape


class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        param = node.pb.reshape_param

        if param.axis != 0:
            log.error('The operation "Reshape" has attribute "axis" with unsupported value "{}"'.format(param['axis']))
            return False

        if param.num_axes != -1:
            log.error('The operation "Reshape" has attribute "num_axes" with unsupported value "{}"'.format(
                param['num_axes']))
            return False

        Reshape.update_node_stat(node, {
            'dim': list(param.shape.dim),
        })
        return cls.enabled
