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


def factor_update(factor: float, real_factor: list, in_shape: list, out_shape: list, name: str):
    """ Updates factor value for layers related to image resizing such as Resample and Interp. """
    if factor is None:
        if real_factor[0] != real_factor[1]:
            log.warning(
                'Cannot deduce a single zoom factor for both height and widths for node {}: [{},{}]/[{},{}] = [{},{}]. '
                'This model will not reshape in IE.'.format(
                    name,
                    out_shape[0],
                    out_shape[1],
                    in_shape[0],
                    in_shape[1],
                    real_factor[0],
                    real_factor[1]
                )
            )
        else:
            factor = real_factor[0]
    return factor
