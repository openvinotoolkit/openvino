// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_fusion.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "op/mha.hpp"

#include "itt.hpp"

// TODO: draw pattern
ov::intel_cpu::MHAFusion::MHAFusion() {
    MATCHER_SCOPE(MHAFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(transpose1, in2);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, mul);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(add, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(reshape1, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto mul_in1 = pattern_to_output.at(in2);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        // TODO: check transpose order

        auto const_node = std::dynamic_pointer_cast<ngraph::opset3::Constant>(mul_in1.get_node_shared_ptr());
        if (!const_node)
            return false;

        std::vector<float> mul_scales;
        if (auto mul_node = std::dynamic_pointer_cast<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = std::dynamic_pointer_cast<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();
        } else {
            return false;
        }

        auto matmul0_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto reshape0_node = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape0_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            // TODO: add valid condition based on reshape pattern
            auto reshape0_pattern_values = reshape0_pattern->cast_vector<int64_t>();
        } else {
            return false;
        }

        auto reshape1_node = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr());
        if (!reshape1_node)
            return false;

        if (auto reshape1_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(in7).get_node_shared_ptr())) {
            // TODO: add valid condition based on reshape pattern
            auto reshape1_pattern_values = reshape1_pattern->cast_vector<int64_t>();
        } else {
            return false;
        }

        if (reshape0_node->get_output_partial_shape(0).rank() != 2 || reshape0_node->get_input_partial_shape(0) != reshape1_node->get_output_partial_shape(0))
            return false;

        auto softmax_node = std::dynamic_pointer_cast<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        auto matmul1_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            transpose3_node->output(0).get_element_type());
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
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

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::MHAFusion2::MHAFusion2() {
    MATCHER_SCOPE(MHAFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto add = std::make_shared<ngraph::opset4::Add>(matmul0, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(matmul1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        // TODO: check transpose order

        auto matmul0_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        auto softmax_node = std::dynamic_pointer_cast<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        auto matmul1_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, std::vector<float>(), false,
                                                            transpose3_node->output(0).get_element_type());
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
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

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

static std::vector<float> simplifyToScale(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fq_node) {
    auto levels = fq_node->get_levels();
    auto input_low = std::dynamic_pointer_cast<ngraph::opset4::Constant>(fq_node->get_input_node_shared_ptr(1))->cast_vector<float>();
    auto input_high = std::dynamic_pointer_cast<ngraph::opset4::Constant>(fq_node->get_input_node_shared_ptr(2))->cast_vector<float>();
    auto output_low = std::dynamic_pointer_cast<ngraph::opset4::Constant>(fq_node->get_input_node_shared_ptr(3))->cast_vector<float>();
    auto output_high = std::dynamic_pointer_cast<ngraph::opset4::Constant>(fq_node->get_input_node_shared_ptr(4))->cast_vector<float>();

    std::vector<float> cl, ch, isc, ish, osc, osh;
    for (int i = 0; i < input_low.size(); i++) {
        cl.push_back(input_low[i]);
    }
    for (int i = 0; i < input_high.size(); i++) {
        ch.push_back(input_high[i]);
    }

    for (int i = 0; i < std::max(input_low.size(), input_high.size()); i++) {
        float il = input_low[input_low.size() == 1 ? 0 : i];
        float ih = input_high[input_high.size() == 1 ? 0 : i];

        isc.push_back((levels - 1) / (ih - il));
        ish.push_back(-il * (levels - 1) / (ih - il));
    }

    for (int i = 0; i < std::max(output_low.size(), output_high.size()); i++) {
        float ol = output_low[output_low.size() == 1 ? 0 : i];
        float oh = output_high[output_high.size() == 1 ? 0 : i];

        osc.push_back((oh - ol) / (levels - 1));
        osh.push_back(ol);
    }

    std::vector<float> outScale;

    if (fq_node->get_output_element_type(0) == ngraph::element::u8 &&
            std::all_of(cl.cbegin(), cl.cend(), [](float val) { return val == 0.0f; }) &&
            std::all_of(ish.cbegin(), ish.cend(), [](float val) { return val == 0.0f; }) &&
            std::all_of(osc.cbegin(), osc.cend(), [](float val) { return val == 1.0f; }) &&
            std::all_of(osh.cbegin(), osh.cend(), [](float val) { return val == 0.0f; })) {
        outScale = isc;
    }

    if (fq_node->get_output_element_type(0) == ngraph::element::i8 &&
            std::all_of(ish.cbegin(), ish.cend(), [](float val) { return std::abs(val - 128.f) < 0.0001f; }) &&
            std::all_of(osc.cbegin(), osc.cend(), [](float val) { return val == 1.f; }) &&
            std::all_of(osh.cbegin(), osh.cend(), [](float val) { return std::abs(val + 128.f) < 0.0001f; })) {
        bool isCropAligned = true;
        for (int i = 0; i < std::max(cl.size(), isc.size()); i++) {
            if (std::abs(cl[cl.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] + 128.f) > 0.0001f) {
                isCropAligned = false;
            }
        }

        for (int i = 0; i < std::max(ch.size(), isc.size()); i++) {
            if (std::abs(ch[ch.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] - 127.f) > 0.0001f) {
                isCropAligned = false;
            }
        }

        if (isCropAligned) {
            outScale = isc;
        }
    }

    return outScale;
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion::MHAQuantFusion() {
    MATCHER_SCOPE(MHAQuantFusion);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in6 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in7 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, transpose1);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul0,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto add = std::make_shared<ngraph::opset4::Add>(fakeQuantize0, in3);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(add, in2);
    auto reshape0 = std::make_shared<ngraph::opset1::Reshape>(mul, in6, true);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(reshape0);
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(softmax, in7, true);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({reshape1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(fakeQuantize1, transpose2);
    auto fakeQuantize2 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize2, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        // todo: check mul shape
        std::vector<float> mul_scales;
        if (auto mul_node = std::dynamic_pointer_cast<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = std::dynamic_pointer_cast<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();
        } else {
            return false;
        }

        // TODO: check transpose order
        auto matmul0_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr())) {
            fq0_scale = simplifyToScale(fq_node);
            if (!fq0_scale.size())
                return false;
        }

        auto reshape0_node = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape0).get_node_shared_ptr());
        if (!reshape0_node)
            return false;

        if (auto reshape0_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(in6).get_node_shared_ptr())) {
            // TODO: add valid condition based on reshape pattern
            auto reshape0_pattern_values = reshape0_pattern->cast_vector<int64_t>();
        } else {
            return false;
        }

        auto reshape1_node = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr());
        if (!reshape1_node)
            return false;

        if (auto reshape1_pattern = std::dynamic_pointer_cast<ngraph::opset4::Constant>(pattern_to_output.at(in7).get_node_shared_ptr())) {
            // TODO: add valid condition based on reshape pattern
            auto reshape1_pattern_values = reshape1_pattern->cast_vector<int64_t>();
        } else {
            return false;
        }

        if (reshape0_node->get_output_partial_shape(0).rank() != 2 || reshape0_node->get_input_partial_shape(0) != reshape1_node->get_output_partial_shape(0))
            return false;

        auto softmax_node = std::dynamic_pointer_cast<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 1)
            return false;

        std::vector<float> fq1_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq1_scale = simplifyToScale(fq_node);
            if (!fq1_scale.size())
                return false;
        }

        auto matmul1_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        std::vector<float> fq2_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize2).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size())
                return false;
        }

        bool is_mul_first = false;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            std::vector<float>(), fq0_scale, fq1_scale, fq2_scale,
                                                            transpose3_node->output(0).get_element_type());
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
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

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}

// TODO: draw pattern
ov::intel_cpu::MHAQuantFusion2::MHAQuantFusion2() {
    MATCHER_SCOPE(MHAQuantFusion2);

    auto in0 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in1 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in2 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in3 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in4 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in5 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in8 = ngraph::pattern::any_input(ngraph::pattern::has_static_shape());
    auto in9 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto in10 = ngraph::pattern::wrap_type<ngraph::opset4::Constant>();
    auto transpose0 = std::make_shared<ngraph::opset3::Transpose>(in0, in4);
    auto transpose1 = std::make_shared<ngraph::opset3::Transpose>(in1, in5);
    auto fakeQuantize0 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({transpose1,
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto matmul0 = std::make_shared<ngraph::opset3::MatMul>(transpose0, fakeQuantize0);
    auto mul = std::make_shared<ngraph::opset3::Multiply>(matmul0, in2);
    auto add = std::make_shared<ngraph::opset4::Add>(mul, in3);
    auto softmax = std::make_shared<ngraph::opset1::Softmax>(add);
    auto transpose2 = std::make_shared<ngraph::opset3::Transpose>(in8, in9);
    auto matmul1 = std::make_shared<ngraph::opset3::MatMul>(softmax, transpose2);
    auto fakeQuantize1 = ngraph::pattern::wrap_type<ngraph::opset1::FakeQuantize>({matmul1,
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>(),
                                                                                   ngraph::pattern::wrap_type<ngraph::opset4::Constant>()});
    auto transpose3 = std::make_shared<ngraph::opset3::Transpose>(fakeQuantize1, in10);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto transpose0_in = pattern_to_output.at(in0);
        auto transpose1_in = pattern_to_output.at(in1);
        auto add_in1 = pattern_to_output.at(in3);
        auto transpose2_in = pattern_to_output.at(in8);

        // todo: check mul shape
        std::vector<float> mul_scales;
        if (auto mul_node = std::dynamic_pointer_cast<ngraph::opset3::Multiply>(pattern_to_output.at(mul).get_node_shared_ptr())) {
            mul_scales = std::dynamic_pointer_cast<ngraph::opset4::Constant>(mul_node->get_input_node_shared_ptr(1))->cast_vector<float>();
        } else {
            return false;
        }

        // TODO: check transpose order
        auto matmul0_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul0).get_node_shared_ptr());
        if (!matmul0_node)
            return false;
        if (matmul0_node->get_transpose_a() || matmul0_node->get_transpose_b())
            return false;

        std::vector<float> fq0_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize0).get_node_shared_ptr())) {
            fq0_scale = simplifyToScale(fq_node);
            if (!fq0_scale.size())
                return false;
        }

        auto softmax_node = std::dynamic_pointer_cast<ngraph::opset1::Softmax>(pattern_to_output.at(softmax).get_node_shared_ptr());
        if (!softmax_node)
            return false;
        if (softmax_node->get_axis() != 3)
            return false;

        std::vector<float> fq1_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq1_scale = simplifyToScale(fq_node);
            if (!fq1_scale.size())
                return false;
        }

        auto matmul1_node = std::dynamic_pointer_cast<ngraph::opset3::MatMul>(pattern_to_output.at(matmul1).get_node_shared_ptr());
        if (!matmul1_node)
            return false;
        if (matmul1_node->get_transpose_a() || matmul1_node->get_transpose_b())
            return false;

        std::vector<float> fq2_scale;
        if (auto fq_node = std::dynamic_pointer_cast<ngraph::opset1::FakeQuantize>(pattern_to_output.at(fakeQuantize1).get_node_shared_ptr())) {
            fq2_scale = simplifyToScale(fq_node);
            if (!fq2_scale.size())
                return false;
        }

        bool is_mul_first = true;
        auto transpose3_node = pattern_to_output.at(transpose3).get_node_shared_ptr();
        auto mha = std::make_shared<ov::intel_cpu::MHANode>(transpose0_in, transpose1_in, add_in1, transpose2_in, mul_scales, is_mul_first,
                                                            fq0_scale, std::vector<float>(), std::vector<float>(), fq1_scale,
                                                            transpose3_node->output(0).get_element_type());
        mha->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(transpose0).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize0).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul0).get_node_shared_ptr(),
                                   pattern_to_output.at(mul).get_node_shared_ptr(),
                                   pattern_to_output.at(add).get_node_shared_ptr(),
                                   pattern_to_output.at(softmax).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(matmul1).get_node_shared_ptr(),
                                   pattern_to_output.at(fakeQuantize1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                  },
                                  mha);

        if (transformation_callback(mha)) {
            return false;
        }

        ngraph::replace_node(m.get_match_root(), mha);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose3, matcher_name);
    this->register_matcher(m, callback);
}
