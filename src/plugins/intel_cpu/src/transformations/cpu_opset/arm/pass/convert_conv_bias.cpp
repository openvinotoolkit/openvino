// Copyright (C) 2020-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_conv_bias.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/rt_info/dequantization_node.hpp"

using namespace ov::pass;

ov::intel_cpu::ConvertConvolutionBias::ConvertConvolutionBias() {
    auto conv_i8_activation = pattern::any_input(pattern::type_matches(element::i8));
    auto conv_i8_weights = pattern::any_input(pattern::type_matches(element::i8));
    auto conv_i8 = pattern::wrap_type<ov::op::v1::Convolution>({conv_i8_activation, conv_i8_weights});

    auto conv_u8_activation = pattern::any_input(pattern::type_matches(element::u8));
    auto conv_i8_u8_weights = pattern::any_input(pattern::type_matches_any({element::i8, element::u8}));
    auto conv_u8 = pattern::wrap_type<ov::op::v1::Convolution>({conv_u8_activation, conv_i8_u8_weights});
    auto conv_m = conv_u8 | conv_i8;

    auto bias_const_m = pattern::wrap_type<ov::op::v0::Constant>([](ov::Output<ov::Node> output) {
        return !pattern::type_matches(ov::element::i32)(output);
    });
    auto add_m = pattern::wrap_type<ov::op::v1::Add>({conv_m, bias_const_m});
    auto multiply_m = pattern::wrap_type<ov::op::v1::Multiply>({add_m, pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
        if (!mul) {
            return false;
        }
        // mark Multiply as dequantization node to avoid its conversion to PowerStatic
        ov::mark_as_dequantization_node(mul);

        const auto& pattern_map = m.get_pattern_value_map();
        auto bias_const = pattern_map.at(bias_const_m).get_node_shared_ptr();
        auto round = std::make_shared<ov::op::v5::Round>(bias_const, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        auto convert_to_i32 = std::make_shared<ov::op::v0::Convert>(round, ov::element::i32);

        auto add = pattern_map.at(add_m).get_node_shared_ptr();
        auto new_add = std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Add>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(pattern_map.at(conv_m), ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(convert_to_i32->output(0), ov::element::f32).get());
        new_add->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({add, bias_const}, {round, convert_to_i32, new_add});
        ov::replace_node(add, new_add);

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(multiply_m, "ConvertConvolutionBias");
    register_matcher(matcher, callback);
}