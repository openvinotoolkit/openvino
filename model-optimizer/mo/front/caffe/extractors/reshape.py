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
import logging as log

from mo.front.common.partial_infer.elemental import single_output_infer
from mo.front.common.partial_infer.reshape import tf_reshape_shape_infer


def reshape_ext(pl, ml):
    param = pl.reshape_param

    attrs = {
        'op': 'Reshape',
        'type': 'Reshape',
        'axis': param.axis,
        'num_axes': param.num_axes,
        'dim': list(param.shape.dim),
        'infer': lambda node: single_output_infer(node, tf_reshape_shape_infer)
    }
    if attrs['axis'] != 0:
        log.error('The operation "Reshape" has attribute "axis" with unsupported value "{}"'.format(attrs['axis']))
        return None
    if attrs['num_axes'] != -1:
        log.error('The operation "Reshape" has attribute "num_axes" with unsupported value "{}"'.format(
            attrs['num_axes']))
        return None

    return attrs
