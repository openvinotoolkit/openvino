# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp, CaffePythonFrontExtractorOp


class PythonFrontExtractorOp(FrontExtractorOp):
    op = 'Python'
    enabled = True

    @classmethod
    def extract(cls, node):
        module = node.pb.python_param.module
        layer = node.pb.python_param.layer
        layer_type = '{}.{}'.format(module, layer)
        if layer_type and layer_type in CaffePythonFrontExtractorOp.registered_ops:
            if hasattr(CaffePythonFrontExtractorOp.registered_ops[layer_type], 'extract'):
                # CaffePythonFrontExtractorOp.registered_ops[layer_type] is object of FrontExtractorOp and has the
                # function extract
                return CaffePythonFrontExtractorOp.registered_ops[layer_type].extract(node)
            else:
                # User defined only Op for this layer and CaffePythonFrontExtractorOp.registered_ops[layer_type] is
                # special extractor for Op
                return CaffePythonFrontExtractorOp.registered_ops[layer_type](node)
