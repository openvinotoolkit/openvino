# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.einsum import Einsum
from mo.front.extractor import FrontExtractorOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class EinsumExtractor(FrontExtractorOp):
    op = '_npi_einsum'
    enabled = True

    @classmethod
    def extract(cls, einsum_node):
        einsum_name = einsum_node.soft_get('name', einsum_node.id)
        attrs = get_mxnet_layer_attrs(einsum_node.symbol_dict)
        equation = attrs.str('subscripts')
        normalized_equation = Einsum.normalize_equation(einsum_name, equation)
        Einsum.update_node_stat(einsum_node, {'equation': normalized_equation})
        return cls.enabled
