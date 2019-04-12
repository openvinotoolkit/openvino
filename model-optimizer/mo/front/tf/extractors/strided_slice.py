"""
 Copyright (c) 2018-2019 Intel Corporation

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
import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


def int_to_array_bit_mask(im):
    list_repr = list(np.binary_repr(im))
    list_repr.reverse()
    list_repr = [int(li) for li in list_repr]
    return np.array(list_repr, dtype=np.int32)


class StridedSliceFrontExtractor(FrontExtractorOp):
    op = 'StridedSlice'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.pb
        bm = int_to_array_bit_mask(pb.attr["begin_mask"].i)
        bm = np.array([1 - b for b in bm], dtype=np.int32)
        em = int_to_array_bit_mask(pb.attr["end_mask"].i)
        em = np.array([1 - b for b in em], dtype=np.int32)
        attrs = {
            'begin_mask': bm,
            'end_mask': em,
            'ellipsis_mask': int_to_array_bit_mask(pb.attr["ellipsis_mask"].i),
            'new_axis_mask': int_to_array_bit_mask(pb.attr["new_axis_mask"].i),
            'shrink_axis_mask': int_to_array_bit_mask(pb.attr["shrink_axis_mask"].i),
        }

        Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)
        return __class__.enabled
