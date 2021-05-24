# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.pad import TFPad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node)
        return cls.enabled


class PadV2FrontExtractor(FrontExtractorOp):
    op = 'PadV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node)
        return cls.enabled


class MirrorPadFrontExtractor(FrontExtractorOp):
    op = 'MirrorPad'
    enabled = True

    @classmethod
    def extract(cls, node):
        TFPad.update_node_stat(node, {'mode': node.pb.attr['mode'].s.decode('utf-8').lower()})
        return cls.enabled
