# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.einsum import Einsum
from mo.front.extractor import FrontExtractorOp


class EinsumExtractor(FrontExtractorOp):
    op = 'Einsum'
    enabled = True

    @classmethod
    def extract(cls, einsum_node):
        einsum_name = einsum_node.soft_get('name', einsum_node.id)
        equation = einsum_node.pb.attr['equation'].s.decode('utf-8')
        normalized_equation = Einsum.normalize_equation(einsum_name, equation)
        Einsum.update_node_stat(einsum_node, {'equation': normalized_equation})
        return cls.enabled
