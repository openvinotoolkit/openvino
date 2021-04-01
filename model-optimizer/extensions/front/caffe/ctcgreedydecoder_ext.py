# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class CTCGreedyDecoderFrontExtractor(FrontExtractorOp):
    op = 'CTCGreedyDecoder'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.ctc_decoder_param

        update_attrs = {
            'ctc_merge_repeated': (int)(param.ctc_merge_repeated)
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        CTCGreedyDecoderOp.update_node_stat(node, mapping_rule)
        return cls.enabled
