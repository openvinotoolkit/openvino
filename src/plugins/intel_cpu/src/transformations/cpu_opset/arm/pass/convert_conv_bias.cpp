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
    auto conv_mul_add_fq = std::make_shared<ov::intel_cpu::ConvMulAddFQBlock>(true);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out = conv_mul_add_fq->get_anchor("convolution", pattern_map);
        const auto mul_out = conv_mul_add_fq->get_anchor("multiply", pattern_map);
        const auto add_out = conv_mul_add_fq->get_anchor("add", pattern_map);
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

        if (fakeQuantize->get_output_element_type(0) != conv->get_input_element_type(0)) {
            return false;
        }
        auto new_mul = ov::as_type_ptr<ov::opset1::Multiply>(
            low_precision::NetworkHelper::swapMultiplyAndAdd(ov::as_type_ptr<ov::opset1::Add>(add), 0));
        if (!new_mul) {
            return false;
        }
        // mark Multiply as dequantization node to avoid its conversion to PowerStatic
        ov::mark_as_dequantization_node(new_mul);

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

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(conv_mul_add_fq, "ConvertConvolutionBias");
    register_matcher(matcher, callback);
}
