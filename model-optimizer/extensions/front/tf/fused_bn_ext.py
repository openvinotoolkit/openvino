"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from extensions.ops.BatchNormInference import BatchNormInference
from extensions
from extensions.ops.

'''
class FusedBatchNormBaseExtractor(FrontExtractorOp):
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        is_training = pb.attr['is_training'].b
        attrs = {
            'data_format': pb.attr["data_format"].s,
            'data_type': tf_dtype_extractor(pb.attr["T"].type),
            'eps': pb.attr['epsilon'].f,
            'is_training': is_training
        }
        
        
        
        
class FusedBatchNormExtractor(FrontExtractorOp):
    op = "FusedBatchNorm"
    
    def __init__(self):
        super().__init__(self)
'''