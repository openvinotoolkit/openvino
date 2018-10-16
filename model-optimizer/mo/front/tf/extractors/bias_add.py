"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer


def tf_bias_add_ext(pb):
    result = {
        'infer': copy_shape_infer
    }
    # TF documentation is unclear whether broadcast works along C dimension or along lower dimension
    # independently of data_format. So we replace it by Add only in case of C is lower dimension,
    # and then the regular broadcast semantics works correctly for sure.
    if pb.attr['data_format'].s == b"NHWC":
        result['op'] = 'Add'
        result['can_be_bias'] = True
    return result
