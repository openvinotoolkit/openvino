# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.group_norm import GroupNorm


class ExperimentalDetectronGroupNorm(FrontExtractorOp):
    op = 'ExperimentalDetectronGroupNorm'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'eps': np.array(onnx_attr(node, 'eps', 'f', default=1e-6), dtype=np.float),
            'num_groups': np.array(onnx_attr(node, 'num_groups', 'i', default=1), dtype=np.int64),
        }
        GroupNorm.update_node_stat(node, attrs)
        return cls.enabled


class GroupNormExtractor(FrontExtractorOp):
    op = 'GroupNorm'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'eps': np.array(onnx_attr(node, 'eps', 'f', default=1e-6), dtype=np.float),
            'num_groups': np.array(onnx_attr(node, 'num_groups', 'i', default=1), dtype=np.int64),
        }
        GroupNorm.update_node_stat(node, attrs)
        return cls.enabled
