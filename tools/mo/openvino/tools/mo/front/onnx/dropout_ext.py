# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.identity import Identity
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.utils.error import Error


class DropoutFrontExtractor(FrontExtractorOp):
    op = 'Dropout'
    enabled = True

    @classmethod
    def extract(cls, node):
        # some Dropout flavors doesn't have is_test attribute; when it is missing, interpret it as 1
        is_test = onnx_attr(node, 'is_test', 'i', 1)
        if len(node.out_nodes()) > 1:
            raise Error('Dropout node {} has more than one consumer. Unsupported.', node.name)
        if not is_test:
            raise Error('Dropout node {} has is_test: 0. This means training mode which is not supported.', node.name)
        Identity.update_node_stat(node)
        return cls.enabled
