# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.einsum import Einsum
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class EinsumExtractor(FrontExtractorOp):
    op = 'Einsum'
    enabled = True

    @classmethod
    def extract(cls, einsum_node):
        einsum_name = einsum_node.soft_get('name', einsum_node.id)
        equation = onnx_attr(einsum_node, 'equation', 's').decode(encoding="utf-8")
        normalized_equation = Einsum.normalize_equation(einsum_name, equation)
        Einsum.update_node_stat(einsum_node, {'equation': normalized_equation})
        return cls.enabled
