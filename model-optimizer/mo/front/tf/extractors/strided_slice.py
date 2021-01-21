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
import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.strided_slice import StridedSlice


def int_to_array_bit_mask(im):
    list_repr = list(np.binary_repr(im))
    list_repr.reverse()
    list_repr = [int(li) for li in list_repr]
    return np.array(list_repr, dtype=np.int32)


class StridedSliceFrontExtractor(FrontExtractorOp):
    op = 'StridedSlice'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.pb

        attrs = {
            'begin_mask': int_to_array_bit_mask(pb.attr["begin_mask"].i),
            'end_mask': int_to_array_bit_mask(pb.attr["end_mask"].i),
            'ellipsis_mask': int_to_array_bit_mask(pb.attr["ellipsis_mask"].i),
            'new_axis_mask': int_to_array_bit_mask(pb.attr["new_axis_mask"].i),
            'shrink_axis_mask': int_to_array_bit_mask(pb.attr["shrink_axis_mask"].i),
        }
        attrs['begin_mask'] = np.array([1 - b for b in attrs['begin_mask']], dtype=np.int32)
        attrs['end_mask'] = np.array([1 - b for b in attrs['end_mask']], dtype=np.int32)

        dims = max(map(lambda x: len(x[1]), attrs.items()))

        def extend_mask(in_mask, dims=dims, zeros=True):
            mask = list(in_mask)
            if len(mask) < dims:
                if zeros:
                    mask.extend(np.zeros(dims - len(mask), dtype=np.int32))
                else:
                    mask.extend(np.ones(dims - len(mask), dtype=np.int32))
            return np.array(mask, dtype=np.int32)

        attrs['begin_mask'] = extend_mask(attrs['begin_mask'], zeros=False)
        attrs['end_mask'] = extend_mask(attrs['end_mask'], zeros=False)
        attrs['new_axis_mask'] = extend_mask(attrs['new_axis_mask'])
        attrs['shrink_axis_mask'] = extend_mask(attrs['shrink_axis_mask'])
        attrs['ellipsis_mask'] = extend_mask(attrs['ellipsis_mask'])

        StridedSlice.update_node_stat(node, attrs)
        return cls.enabled
