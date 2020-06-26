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
import pytest

import ngraph as ng


@pytest.fixture()
def _proposal_node():
    attributes = {
        "attrs.base_size": np.uint16(1),
        "attrs.pre_nms_topn": np.uint16(20),
        "attrs.post_nms_topn": np.uint16(64),
        "attrs.nms_thresh": np.float64(0.34),
        "attrs.feat_stride": np.uint16(16),
        "attrs.min_size": np.uint16(32),
        "attrs.ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=np.float64),
        "attrs.scale": np.array([2, 3, 3, 4], dtype=np.float64),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], np.float64, "class_probs")
    class_logits = ng.parameter([batch_size, 24, 34, 62], np.float64, "class_logits")
    image_shape = ng.parameter([3], np.float64, "image_shape")
    return ng.proposal(class_probs, class_logits, image_shape, attributes)


def test_dynamic_attributes_softmax():
    axis = 2
    data = ng.parameter([1, 2, 3, 4], np.float32, "data_in")
    node = ng.softmax(data, axis)

    assert node.get_axis() == axis
    node.set_axis(3)
    assert node.get_axis() == 3


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_dynamic_get_attribute_value(int_dtype, fp_dtype):
    attributes = {
        "attrs.num_classes": int_dtype(85),
        "attrs.background_label_id": int_dtype(13),
        "attrs.top_k": int_dtype(16),
        "attrs.variance_encoded_in_target": True,
        "attrs.keep_top_k": np.array([64, 32, 16, 8], dtype=int_dtype),
        "attrs.code_type": "pytorch.some_parameter_name",
        "attrs.share_location": False,
        "attrs.nms_threshold": fp_dtype(0.645),
        "attrs.confidence_threshold": fp_dtype(0.111),
        "attrs.clip_after_nms": True,
        "attrs.clip_before_nms": False,
        "attrs.decrease_label_id": True,
        "attrs.normalized": True,
        "attrs.input_height": int_dtype(86),
        "attrs.input_width": int_dtype(79),
        "attrs.objectness_score": fp_dtype(0.77),
    }

    box_logits = ng.parameter([4, 1, 5, 5], fp_dtype, "box_logits")
    class_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "class_preds")
    proposals = ng.parameter([2, 1, 4, 5], fp_dtype, "proposals")
    aux_class_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "aux_class_preds")
    aux_box_preds = ng.parameter([2, 1, 4, 5], fp_dtype, "aux_box_preds")

    node = ng.detection_output(
        box_logits, class_preds, proposals, attributes, aux_class_preds, aux_box_preds
    )

    assert node.get_num_classes() == int_dtype(85)
    assert node.get_background_label_id() == int_dtype(13)
    assert node.get_top_k() == int_dtype(16)
    assert node.get_variance_encoded_in_target() == True
    assert np.all(np.equal(node.get_keep_top_k(), np.array([64, 32, 16, 8], dtype=int_dtype)))
    assert node.get_code_type() == "pytorch.some_parameter_name"
    assert node.get_share_location() == False
    assert np.isclose(node.get_nms_threshold(), fp_dtype(0.645))
    assert np.isclose(node.get_confidence_threshold(), fp_dtype(0.111))
    assert node.get_clip_after_nms() == True
    assert node.get_clip_before_nms() == False
    assert node.get_decrease_label_id() == True
    assert node.get_normalized() == True
    assert node.get_input_height() == int_dtype(86)
    assert node.get_input_width() == int_dtype(79)
    assert np.isclose(node.get_objectness_score(), fp_dtype(0.77))
    assert node.get_num_classes() == int_dtype(85)


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.uint32, np.float16),
        (np.uint32, np.float64),
    ],
)
def test_dynamic_set_attribute_value(int_dtype, fp_dtype):
    attributes = {
        "attrs.base_size": int_dtype(1),
        "attrs.pre_nms_topn": int_dtype(20),
        "attrs.post_nms_topn": int_dtype(64),
        "attrs.nms_thresh": fp_dtype(0.34),
        "attrs.feat_stride": int_dtype(16),
        "attrs.min_size": int_dtype(32),
        "attrs.ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=fp_dtype),
        "attrs.scale": np.array([2, 3, 3, 4], dtype=fp_dtype),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], fp_dtype, "class_probs")
    class_logits = ng.parameter([batch_size, 24, 34, 62], fp_dtype, "class_logits")
    image_shape = ng.parameter([3], fp_dtype, "image_shape")
    node = ng.proposal(class_probs, class_logits, image_shape, attributes)

    node.set_base_size(int_dtype(15))
    node.set_pre_nms_topn(int_dtype(7))
    node.set_post_nms_topn(int_dtype(33))
    node.set_nms_thresh(fp_dtype(1.55))
    node.set_feat_stride(int_dtype(8))
    node.set_min_size(int_dtype(123))
    node.set_ratio(np.array([1.1, 2.5, 3.0, 4.5], dtype=fp_dtype))
    node.set_scale(np.array([2.1, 3.2, 3.3, 4.4], dtype=fp_dtype))
    node.set_clip_before_nms(True)
    node.set_clip_after_nms(True)
    node.set_normalize(True)
    node.set_box_size_scale(fp_dtype(1.34))
    node.set_box_coordinate_scale(fp_dtype(0.88))
    node.set_framework("OpenVINO")

    assert node.get_base_size() == int_dtype(15)
    assert node.get_pre_nms_topn() == int_dtype(7)
    assert node.get_post_nms_topn() == int_dtype(33)
    assert np.isclose(node.get_nms_thresh(), fp_dtype(1.55))
    assert node.get_feat_stride() == int_dtype(8)
    assert node.get_min_size() == int_dtype(123)
    assert np.allclose(node.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=fp_dtype))
    assert np.allclose(node.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=fp_dtype))
    assert node.get_clip_before_nms() == True
    assert node.get_clip_after_nms() == True
    assert node.get_normalize() == True
    assert np.isclose(node.get_box_size_scale(), fp_dtype(1.34))
    assert np.isclose(node.get_box_coordinate_scale(), fp_dtype(0.88))
    assert node.get_framework() == "OpenVINO"


def test_dynamic_attr_cache(_proposal_node):
    node = _proposal_node

    assert not node._attr_cache_valid
    node.set_nms_thresh(1.3453678102)
    assert not node._attr_cache_valid
    assert np.isclose(node.get_nms_thresh(), np.float64(1.3453678102))
    assert node._attr_cache_valid


def test_dynamic_attr_transitivity(_proposal_node):
    node = _proposal_node
    node2 = node

    node.set_ratio(np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))
    assert np.allclose(node.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))
    assert np.allclose(node2.get_ratio(), np.array([1.1, 2.5, 3.0, 4.5], dtype=np.float64))

    node2.set_scale(np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))
    assert np.allclose(node2.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))
    assert np.allclose(node.get_scale(), np.array([2.1, 3.2, 3.3, 4.4], dtype=np.float64))


def test_dynamic_attributes_simple():
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    W_shape = [3 * hidden_size, input_size]
    R_shape = [3 * hidden_size, hidden_size]
    B_shape = [4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    activations = ["tanh", "relu"]
    activations_alpha = [1.0, 2.0]
    activations_beta = [1.0, 2.0]
    clip = 0.5
    linear_before_reset = True

    node = ng.gru_cell(
        parameter_X,
        parameter_H_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activations_alpha,
        activations_beta,
        clip,
        linear_before_reset,
    )

    assert node.get_hidden_size() == hidden_size
    assert all(map(lambda x, y: x == y, node.get_activations(), activations))
    assert all(np.equal(node.get_activations_alpha(), activations_alpha))
    assert all(np.equal(node.get_activations_beta(), activations_beta))
    assert node.get_linear_before_reset() == linear_before_reset
    assert np.isclose(node.get_clip(), clip)
