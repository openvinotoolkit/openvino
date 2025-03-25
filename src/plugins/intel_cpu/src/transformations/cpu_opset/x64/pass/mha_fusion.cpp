// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "simplify_fakequantize.hpp"
#include "transformations/cpu_opset/x64/op/mha.hpp"
#include "transformations/utils/utils.hpp"

// TODO: draw pattern
ov::intel_cpu::MHAFloatFusion::MHAFloatFusion() {
    MATCHER_SCOPE(MHAFloatFusion);

    auto in0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in2 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in3 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in4 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in5 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in6 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in7 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in8 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in9 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in10 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto transpose0 = std::make_shared<ov::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ov::opset3::Transpose>(in1, in5);
    auto mul = std::make_shared<ov::opset3::Multiply>(transpose1, in2);
    auto matmul0 = std::make_shared<ov::opset3::MatMul>(transpose0, mul);
    auto add = std::make_shared<ov::opset4::Add>(matmul0, in3);
    auto reshape0 = std::make_shared<ov::opset1::Reshape>(add, in6, true);
    auto softmax = std::make_shared<ov::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ov::opset1::Reshape>(softmax, in7, true);
    auto transpose2 = std::make_shared<ov::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ov::opset3::MatMul>(reshape1, transpose2);
    auto transpose3 = std::make_shared<ov::opset3::Transpose>(matmul1, in10);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto mul_in1 = pattern_to_output.at(in2);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() ||
            transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ov::as_type_ptr<ov::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales =
                ov::as_type_ptr<ov::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ov::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        auto matmul0_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node) {
            return false;
        }
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b()) {
            return false;
        }

        auto reshape0_node = ov::as_type_ptr<ov::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node) {
            return false;
        }

        if (auto reshape_pattern =
                ov::as_type_ptr<ov::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {
                static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] * reshape0_node->get_input_shape(0)[1] *
                                     reshape0_node->get_input_shape(0)[2]),
                -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node =
                ov::as_type_ptr<ov::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ov::as_type_ptr<ov::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node) {
            return false;
        }
        if (softmax_node->get_axis() != 1) {
            return false;
        }

        auto matmul1_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node) {
            return false;
        }
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b()) {
            return false;
        }

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in,
                                                            transpose1_in,
                                                            add_in1,
                                                            transpose2_in,
                                                            mul_scales,
                                                            is_mul_first,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(transpose0).get_node_shared_ptr(),
                pattern_to_output.at(transpose1).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
                pattern_to_output.at(matmul0).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(reshape0).get_node_shared_ptr(),
                pattern_to_output.at(softmax).get_node_shared_ptr(),
                pattern_to_output.at(reshape1).get_node_shared_ptr(),
                pattern_to_output.at(transpose2).get_node_shared_ptr(),
                pattern_to_output.at(matmul1).get_node_shared_ptr(),
                pattern_to_output.at(transpose3).get_node_shared_ptr(),
            },
            mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ov::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::MHAFloatFusion2::MHAFloatFusion2() {
    MATCHER_SCOPE(MHAFloatFusion2);

    auto in0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in3 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in4 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in5 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in6 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in7 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in8 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in9 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in10 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto transpose0 = std::make_shared<ov::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ov::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ov::opset3::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<ov::opset4::Add>(matmul0, in3);
    auto softmax = std::make_shared<ov::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ov::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ov::opset3::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<ov::opset3::Transpose>(matmul1, in10);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() ||
            transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }

        auto matmul0_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node) {
            return false;
        }
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b()) {
            return false;
        }

        auto softmax_node = ov::as_type_ptr<ov::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node) {
            return false;
        }
        if (softmax_node->get_axis() != 3) {
            return false;
        }

        auto matmul1_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node) {
            return false;
        }
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b()) {
            return false;
        }

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in,
                                                            transpose1_in,
                                                            add_in1,
                                                            transpose2_in,
                                                            std::vector<float>(),
                                                            false,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(transpose0).get_node_shared_ptr(),
                pattern_to_output.at(transpose1).get_node_shared_ptr(),
                pattern_to_output.at(matmul0).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(softmax).get_node_shared_ptr(),
                pattern_to_output.at(transpose2).get_node_shared_ptr(),
                pattern_to_output.at(matmul1).get_node_shared_ptr(),
                pattern_to_output.at(transpose3).get_node_shared_ptr(),
            },
            mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ov::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion::MHAQuantFusion() {
    MATCHER_SCOPE(MHAQuantFusion);

    auto in0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in2 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in3 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in4 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in5 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in6 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in7 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in8 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in9 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in10 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto transpose0 = std::make_shared<ov::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ov::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ov::opset3::MatMul>(transpose0, transpose1);
    auto fakeQuantize0 =
        ov::pass::pattern::wrap_type<ov::opset1::FakeQuantize>({matmul0,
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>()});
    auto add = std::make_shared<ov::opset4::Add>(fakeQuantize0, in3);
    auto mul = std::make_shared<ov::opset3::Multiply>(add, in2);
    auto reshape0 = std::make_shared<ov::opset1::Reshape>(mul, in6, true);
    auto softmax = std::make_shared<ov::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ov::opset1::Reshape>(softmax, in7, true);
    auto fakeQuantize1 =
        ov::pass::pattern::wrap_type<ov::opset1::FakeQuantize>({reshape1,
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>()});
    auto transpose2 = std::make_shared<ov::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ov::opset3::MatMul>(fakeQuantize1, transpose2);
    auto fakeQuantize2 =
        ov::pass::pattern::wrap_type<ov::opset1::FakeQuantize>({matmul1,
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>()});
    auto transpose3 = std::make_shared<ov::opset3::Transpose>(fakeQuantize2, in10);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() ||
            transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ov::as_type_ptr<ov::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales =
                ov::as_type_ptr<ov::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();
            auto expected_shape = ov::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }

        auto matmul0_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node) {
            return false;
        }
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b()) {
            return false;
        }

        std::vector<float> fq0_scale;
        auto fq0_node =
            ov::as_type_ptr<ov::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size()) {
                return false;
            }
        }

        auto reshape0_node = ov::as_type_ptr<ov::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node) {
            return false;
        }

        if (auto reshape_pattern =
                ov::as_type_ptr<ov::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0).size() != 4) {
                return false;
            }

            std::vector<int64_t> reshapeConstData = {
                static_cast<int64_t>(reshape0_node->get_input_shape(0)[0] * reshape0_node->get_input_shape(0)[1] *
                                     reshape0_node->get_input_shape(0)[2]),
                -1};

            if (reshape_pattern->cast_vector<int64_t>() != reshapeConstData) {
                return false;
            }
        } else {
            return false;
        }

        if (auto reshape1_node =
                ov::as_type_ptr<ov::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr())) {
            if (reshape0_node->get_input_shape(0) != reshape1_node->get_output_shape(0)) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ov::as_type_ptr<ov::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node) {
            return false;
        }
        if (softmax_node->get_axis() != 1) {
            return false;
        }

        std::vector<float> fq1_scale;
        auto fq1_node =
            ov::as_type_ptr<ov::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        if (fq1_node) {
            fq1_scale = simplifyToScale(fq1_node);
            if (!fq1_scale.size()) {
                return false;
            }
        } else {
            return false;
        }

        auto matmul1_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node) {
            return false;
        }
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b()) {
            return false;
        }

        std::vector<float> fq2_scale;
        if (auto fq_node =
                ov::as_type_ptr<ov::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize2).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size()) {
                return false;
            }
        }

        bool is_mul_first = false;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(
            transpose0_in,
            transpose1_in,
            add_in1,
            transpose2_in,
            mul_scales,
            is_mul_first,
            std::vector<float>(),
            fq0_scale,
            fq1_scale,
            fq2_scale,
            ov::element::dynamic,
            fq0_node ? fq0_node->get_output_element_type(0) : ov::element::dynamic,
            fq1_node->get_output_element_type(0),
            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(transpose0).get_node_shared_ptr(),
                pattern_to_output.at(transpose1).get_node_shared_ptr(),
                pattern_to_output.at(matmul0).get_node_shared_ptr(),
                pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
                pattern_to_output.at(reshape0).get_node_shared_ptr(),
                pattern_to_output.at(softmax).get_node_shared_ptr(),
                pattern_to_output.at(reshape1).get_node_shared_ptr(),
                pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                pattern_to_output.at(transpose2).get_node_shared_ptr(),
                pattern_to_output.at(matmul1).get_node_shared_ptr(),
                pattern_to_output.at(fakeQuantize2).get_node_shared_ptr(),
                pattern_to_output.at(transpose3).get_node_shared_ptr(),
            },
            mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ov::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion2::MHAQuantFusion2() {
    MATCHER_SCOPE(MHAQuantFusion2);

    auto in0 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in1 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in2 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in3 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in4 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in5 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in8 = ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape());
    auto in9 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto in10 = ov::pass::pattern::wrap_type<ov::opset4::Constant>();
    auto transpose0 = std::make_shared<ov::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ov::opset3::Transpose>(in1, in5);
    auto fakeQuantize0 =
        ov::pass::pattern::wrap_type<ov::opset1::FakeQuantize>({transpose1,
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>()});
    auto matmul0 = std::make_shared<ov::opset3::MatMul>(transpose0, fakeQuantize0);
    auto mul = std::make_shared<ov::opset3::Multiply>(matmul0, in2);
    auto add = std::make_shared<ov::opset4::Add>(mul, in3);
    auto softmax = std::make_shared<ov::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ov::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ov::opset3::MatMul>(softmax, transpose2);
    auto fakeQuantize1 =
        ov::pass::pattern::wrap_type<ov::opset1::FakeQuantize>({matmul1,
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>(),
                                                                ov::pass::pattern::wrap_type<ov::opset4::Constant>()});
    auto in11 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{matmul1, fakeQuantize1});
    auto transpose3 = std::make_shared<ov::opset3::Transpose>(in11, in10);

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        if (transpose0_in.get_shape() != transpose1_in.get_shape() ||
            transpose0_in.get_shape() != transpose2_in.get_shape()) {
            return false;
        }

        if (transpose0_in.get_shape().size() != 4) {
            return false;
        }

        auto expected_add_shape = Shape({transpose0_in.get_shape()[0], 1, 1, transpose0_in.get_shape()[1]});
        if (add_in1.get_shape() != expected_add_shape) {
            return false;
        }

        std::vector<float> mul_scales;
        if (auto mul_node = ov::as_type_ptr<ov::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales =
                ov::as_type_ptr<ov::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();

            auto expected_shape = ov::Shape({1, transpose0_in.get_shape()[2], 1, 1});
            if (mul_scales.size() != 1 && mul_node->get_input_shape(1) != expected_shape) {
                return false;
            }
        } else {
            return false;
        }

        if (!valid_transpose_order(pattern_to_output.at(in4).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in5).get_node_shared_ptr(), {0, 2, 3, 1})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in9).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }
        if (!valid_transpose_order(pattern_to_output.at(in10).get_node_shared_ptr(), {0, 2, 1, 3})) {
            return false;
        }

        auto matmul0_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node) {
            return false;
        }
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b()) {
            return false;
        }

        std::vector<float> fq0_scale;
        auto fq0_node =
            ov::as_type_ptr<ov::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr());
        if (fq0_node) {
            fq0_scale = simplifyToScale(fq0_node);
            if (!fq0_scale.size()) {
                return false;
            }
        } else {
            return false;
        }

        auto softmax_node = ov::as_type_ptr<ov::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node) {
            return false;
        }
        if (softmax_node->get_axis() != 3) {
            return false;
        }

        std::vector<float> fq1_scale;
        const bool fakeQuantize1Exists = pattern_to_output.find(fakeQuantize1) != pattern_to_output.end();
        if (fakeQuantize1Exists) {
            if (auto fq_node = ov::as_type_ptr<ov::opset1::FakeQuantize>(
                    pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
                fq1_scale = simplifyToScale(fq_node);
                if (!fq1_scale.size()) {
                    return false;
                }
            }
        }

        auto matmul1_node = ov::as_type_ptr<ov::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node) {
            return false;
        }
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b()) {
            return false;
        }

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in,
                                                            transpose1_in,
                                                            add_in1,
                                                            transpose2_in,
                                                            mul_scales,
                                                            is_mul_first,
                                                            fq0_scale,
                                                            std::vector<float>(),
                                                            std::vector<float>(),
                                                            fq1_scale,
                                                            fq0_node->get_output_element_type(0),
                                                            ov::element::dynamic,
                                                            ov::element::dynamic,
                                                            transpose3_node->get_output_element_type(0));
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        std::vector<std::shared_ptr<Node>> merged = {
            pattern_to_output.at(transpose0).get_node_shared_ptr(),
            pattern_to_output.at(transpose1).get_node_shared_ptr(),
            pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
            pattern_to_output.at(matmul0).get_node_shared_ptr(),
            pattern_to_output.at(mul).get_node_shared_ptr(),
            pattern_to_output.at(add).get_node_shared_ptr(),
            pattern_to_output.at(softmax).get_node_shared_ptr(),
            pattern_to_output.at(transpose2).get_node_shared_ptr(),
            pattern_to_output.at(matmul1).get_node_shared_ptr(),
            pattern_to_output.at(transpose3).get_node_shared_ptr(),
        };

        if (fakeQuantize1Exists) {
            merged.push_back(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr());
        }
        ov::copy_runtime_info(merged, mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ov::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}
