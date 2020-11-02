# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import ngraph as ng
from ngraph.impl import Shape, Type


def test_proposal_props():
    float_dtype = np.float32
    batch_size = 1
    post_nms_topn = 20
    probs = ng.parameter(Shape([batch_size, 8, 255, 255]), dtype=float_dtype, name="probs")
    deltas = ng.parameter(Shape([batch_size, 16, 255, 255]), dtype=float_dtype, name="bbox_deltas")
    im_info = ng.parameter(Shape([4]), dtype=float_dtype, name="im_info")

    attrs = {
        "base_size": np.uint32(85),
        "pre_nms_topn": np.uint32(10),
        "post_nms_topn": np.uint32(post_nms_topn),
        "nms_thresh": np.float32(0.34),
        "feat_stride": np.uint32(16),
        "min_size": np.uint32(32),
        "ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=np.float32),
        "scale": np.array([2, 3, 3, 4], dtype=np.float32),
    }

    node = ng.proposal(probs, deltas, im_info, attrs)

    assert node.get_type_name() == "Proposal"
    assert node.get_output_size() == 2

    assert list(node.get_output_shape(0)) == [batch_size * post_nms_topn, 5]
    assert list(node.get_output_shape(1)) == [batch_size * post_nms_topn]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.f32
