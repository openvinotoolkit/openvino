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

from extensions.ops.ExtractImagePatches import ExtractImagePatches
from mo.front.common.partial_infer.utils import convert_tf_padding_to_str
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_int_list

class ExtractImagePatchesExtractor(FrontExtractorOp):
    op = 'ExtractImagePatches'
    enabled = True

    @classmethod
    def extract(cls, node):

        attrs = {
            'spatial_dims': int64_array([1, 2]),
            'sizes': tf_int_list(node.pb.attr['ksizes'].list),
            'strides': tf_int_list(node.pb.attr['strides'].list),
            'rates': tf_int_list(node.pb.attr['rates'].list),
            'auto_pad': convert_tf_padding_to_str(node.pb.attr['padding'].s.decode()),
        }

        ExtractImagePatches.update_node_stat(node, attrs)
