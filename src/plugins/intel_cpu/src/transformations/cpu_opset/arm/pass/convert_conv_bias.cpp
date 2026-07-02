// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_conv_bias.hpp"

#include <memory>

#include "conv_mul_add_fq_block.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/round.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

using namespace ov::pass;

ov::intel_cpu::ConvertConvolutionBias::ConvertConvolutionBias() {
    auto conv_mul_add_fq = std::make_shared<ov::intel_cpu::ConvMulAddFQBlock>(false);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out = conv_mul_add_fq->get_anchor("convolution", pattern_map);
        const auto mul_out = conv_mul_add_fq->get_anchor("multiply", pattern_map);
        const auto add_out = conv_mul_add_fq->get_anchor("add", pattern_map);
        const auto activation_out = conv_mul_add_fq->get_anchor("activation", pattern_map);
        const auto fq_out = conv_mul_add_fq->get_anchor("fake_quantize", pattern_map);
        if (!conv_out || !mul_out || !add_out || !fq_out) {
            return false;
        }

        auto fakeQuantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_out->get_node_shared_ptr());
        auto mul = mul_out->get_node_shared_ptr();
        auto conv = conv_out->get_node_shared_ptr();
        auto add = add_out->get_node_shared_ptr();
        if (!fakeQuantize || !mul || !conv || !add) {
            return false;
        }

        const bool has_swish = activation_out.has_value();
        if (!has_swish && fakeQuantize->get_output_element_type(0) != conv->get_input_element_type(0)) {
            return false;
        }
        auto new_mul = ov::as_type_ptr<ov::opset1::Multiply>(
            low_precision::NetworkHelper::swapMultiplyAndAdd(ov::as_type_ptr<ov::opset1::Add>(add), 0));
        if (!new_mul) {
            return false;
        }
        // mark Multiply as dequantization node to avoid its conversion to PowerStatic
        ov::mark_as_dequantization_node(new_mul);

        if (!has_swish) {
            add = ov::as_type_ptr<ov::opset1::Add>(new_mul->get_input_node_shared_ptr(0));
            auto bias_const = ov::as_type_ptr<ov::op::v0::Constant>(add->get_input_node_shared_ptr(1));
            auto round = std::make_shared<ov::op::v5::Round>(bias_const, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
            auto convert_to_i32 = std::make_shared<ov::op::v0::Convert>(round, ov::element::i32);

            auto new_add = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Add>>(
                ov::element::TypeVector{ov::element::f32, ov::element::f32},
                ov::element::TypeVector{ov::element::f32},
                ov::op::TemporaryReplaceOutputType(add->input_value(0), ov::element::f32).get(),
                ov::op::TemporaryReplaceOutputType(convert_to_i32->output(0), ov::element::f32).get());
            new_add->set_friendly_name(add->get_friendly_name());
            ov::copy_runtime_info({add, bias_const}, {round, convert_to_i32, new_add});
            ov::replace_node(add, new_add);
        } else {
            auto new_add = ov::as_type_ptr<ov::opset1::Add>(new_mul->get_input_node_shared_ptr(0));
            auto dq_scale_const = ov::as_type_ptr<ov::op::v0::Constant>(new_mul->get_input_node_shared_ptr(1));
            if (new_add && dq_scale_const) {
                auto bias_div_scale =
                    ov::as_type_ptr<ov::op::v0::Constant>(new_add->get_input_node_shared_ptr(1));
                if (bias_div_scale) {
                    const auto bds_vals = bias_div_scale->cast_vector<float>();
                    const auto scale_vals = dq_scale_const->cast_vector<float>();
                    std::vector<float> restored(bds_vals.size());
                    for (size_t i = 0; i < bds_vals.size(); i++) {
                        const float s = (scale_vals.size() == 1) ? scale_vals[0] : scale_vals[i];
                        restored[i] = bds_vals[i] * s;
                    }
                    auto restored_const = ov::op::v0::Constant::create(
                        ov::element::f32, bias_div_scale->get_shape(), restored);
                    restored_const->set_friendly_name(bias_div_scale->get_friendly_name());
                    ov::copy_runtime_info(bias_div_scale, restored_const);
                    ov::replace_node(bias_div_scale, restored_const);
                }
            }
        }

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(conv_mul_add_fq, "ConvertConvolutionBias");
    register_matcher(matcher, callback);
}
