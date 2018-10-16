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

from mo.front.common.partial_infer.squeeze import tf_squeeze_infer
from mo.front.tf.extractors.utils import tf_int_list


def tf_squeeze_ext(pb):
    return {
        # TODO handle case when squeeze_dims are not in pb
        'type': 'Reshape',
        'squeeze_dims': tf_int_list(pb.attr['squeeze_dims'].list),
        'infer': tf_squeeze_infer
    }
