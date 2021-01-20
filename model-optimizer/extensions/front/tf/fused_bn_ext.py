"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from extensions.ops.BatchNormInference import BatchNormInference
from extensions.ops.BatchNormInferenceMultipleOutputs import BatchNormInferenceMO
from extensions.ops.BatchNormTraining import BatchNormTraining
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor
from mo.graph.graph import Node


def tf_fused_batch_norm_extract(node: Node):
    pb = node.pb
    attrs = {
        'data_format': pb.attr["data_format"].s,
        'data_type': tf_dtype_extractor(pb.attr["T"].type),
        'eps': pb.attr['epsilon'].f
    }
    if pb.attr['is_training'].b:
        BatchNormTraining.update_node_stat(node, attrs)
    elif len(node.out_nodes().items()) > 1:
        BatchNormInferenceMO.update_node_stat(node, attrs)
    else:
        BatchNormInference.update_node_stat(node, attrs)


class FusedBatchNormExtractor(FrontExtractorOp):
    op = "FusedBatchNorm"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        tf_fused_batch_norm_extract(node)
        return cls.enabled


class FusedBatchNormV2Extractor(FrontExtractorOp):
    op = "FusedBatchNormV2"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        tf_fused_batch_norm_extract(node)
        return cls.enabled


class FusedBatchNormV3Extractor(FrontExtractorOp):
    op = "FusedBatchNormV3"
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        tf_fused_batch_norm_extract(node)
        return cls.enabled
