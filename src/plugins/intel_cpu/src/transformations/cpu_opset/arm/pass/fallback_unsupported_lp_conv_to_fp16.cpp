// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fallback_unsupported_lp_conv_to_fp16.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace ov::pass;

namespace {
std::pair<ov::Output<ov::Node>, std::shared_ptr<ov::Node>> ensure_fp16(const ov::Output<ov::Node>& input,
                                                                       const std::string& convert_name,
                                                                       const std::shared_ptr<ov::Node>& rt_source) {
    if (input.get_element_type() == ov::element::f16) {
        return {input, nullptr};
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
    convert->set_friendly_name(convert_name);
    ov::copy_runtime_info(rt_source, convert);
    return {convert->output(0), convert};
}

struct ConvMulAddPattern {
    std::shared_ptr<ov::Node> convolution;
    std::shared_ptr<ov::Node> multiply;
    std::shared_ptr<ov::Node> add;
};

struct ConvMulAddFQPattern {
    std::shared_ptr<ov::Node> convolution;
    std::shared_ptr<ov::Node> multiply;
    std::shared_ptr<ov::Node> add;
    std::shared_ptr<ov::Node> fake_quantize;
};

ConvMulAddPattern create_conv_mul_add_pattern() {
    using namespace ov::pass::pattern;

    auto u8_activation = any_input(type_matches(ov::element::u8));
    auto u8_opt_convert = optional<ov::op::v0::Convert>({u8_activation});
    auto u8_zero_point = any_input();
    auto u8_opt_subtract = optional<ov::op::v1::Subtract>({u8_opt_convert, u8_zero_point});
    auto u8_weights = any_input(type_matches_any({ov::element::i8, ov::element::u8}));
    auto conv_u8 = wrap_type<ov::op::v1::Convolution>({u8_opt_subtract, u8_weights});

    auto i8_activation = any_input(type_matches(ov::element::i8));
    auto i8_opt_convert = optional<ov::op::v0::Convert>({i8_activation});
    auto i8_zero_point = any_input();
    auto i8_opt_subtract = optional<ov::op::v1::Subtract>({i8_opt_convert, i8_zero_point});
    auto i8_weights = any_input(type_matches(ov::element::i8));
    auto conv_i8 = wrap_type<ov::op::v1::Convolution>({i8_opt_subtract, i8_weights});

    auto conv = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{conv_u8, conv_i8});
    auto multiply = wrap_type<ov::op::v1::Multiply>({conv, any_input()});
    auto bias_const = wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
        return !type_matches(ov::element::i32)(output);
    });
    auto add = wrap_type<ov::op::v1::Add>({multiply, bias_const});
    return {conv, multiply, add};
}

ConvMulAddFQPattern create_conv_mul_add_fq_pattern() {
    using namespace ov::pass::pattern;

    const auto conv_mul_add = create_conv_mul_add_pattern();
    auto fake_quantize =
        wrap_type<ov::op::v0::FakeQuantize>({conv_mul_add.add, any_input(), any_input(), any_input(), any_input()});
    return {conv_mul_add.convolution, conv_mul_add.multiply, conv_mul_add.add, fake_quantize};
}

bool fallback_lp_conv_to_fp16(const std::shared_ptr<ov::op::v1::Convolution>& conv,
                              const std::shared_ptr<ov::op::v1::Multiply>& mul,
                              const std::shared_ptr<ov::op::v1::Add>& add) {
    if (!conv || !mul || !add) {
        return false;
    }

    constexpr size_t scales_port = 1;
    const auto rank_value = conv->get_input_shape(1).size();
    auto [activation_f16, activation_convert] =
        ensure_fp16(conv->input_value(0), conv->get_friendly_name() + "/ActivationToFP16", conv);

    auto [weights_f16, weights_convert] =
        ensure_fp16(conv->input_value(1), conv->get_friendly_name() + "/WeightsToFP16", conv);
    auto [scales_f16, scales_convert] =
        ensure_fp16(mul->input_value(scales_port), mul->get_friendly_name() + "/ScalesToFP16", mul);

    std::vector<int64_t> reshape_pattern(static_cast<size_t>(rank_value), 1);
    reshape_pattern[0] = -1;
    auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern);
    auto scales_reshape = std::make_shared<ov::op::v1::Reshape>(scales_f16, reshape_const, true);
    scales_reshape->set_friendly_name(mul->get_friendly_name() + "/ScalesToWeightsShape");

    auto scales_to_weights = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_reshape);
    scales_to_weights->set_friendly_name(mul->get_friendly_name() + "/WeightsScaled");
    auto conv_scaled = conv->clone_with_new_inputs({activation_f16, scales_to_weights->output(0)});
    ov::NodeVector rt_info_sources{conv, mul};
    if (activation_convert) {
        rt_info_sources.push_back(activation_convert);
    }
    if (weights_convert) {
        rt_info_sources.push_back(weights_convert);
    }
    if (scales_convert) {
        rt_info_sources.push_back(scales_convert);
    }
    ov::copy_runtime_info(rt_info_sources, {reshape_const, scales_reshape, scales_to_weights, conv_scaled});

    ov::replace_node(mul, conv_scaled);

    // Keep this matched low-precision Conv->Mul->Add slice in FP16 end-to-end for CPU plugin selection.
    if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(conv_scaled)) {
        type_relaxed->set_origin_input_type(ov::element::f16, 0);
        type_relaxed->set_origin_input_type(ov::element::f16, 1);
        if (add->get_output_element_type(0) == ov::element::f16) {
            type_relaxed->set_overridden_output_type(ov::element::f16, 0);
        }
        conv_scaled->validate_and_infer_types();
    }

    return true;
}

}  // namespace

ov::intel_cpu::FallbackUnsupportedLPConvToFP16::FallbackUnsupportedLPConvToFP16() {
    const auto conv_mul_add_fq = create_conv_mul_add_fq_pattern();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out_it = pattern_map.find(conv_mul_add_fq.convolution);
        const auto mul_out_it = pattern_map.find(conv_mul_add_fq.multiply);
        const auto add_out_it = pattern_map.find(conv_mul_add_fq.add);
        const auto fq_out_it = pattern_map.find(conv_mul_add_fq.fake_quantize);
        if (conv_out_it == pattern_map.end() || mul_out_it == pattern_map.end() ||
            add_out_it == pattern_map.end() || fq_out_it == pattern_map.end()) {
            return false;
        }

        const auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(conv_out_it->second.get_node_shared_ptr());
        const auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(mul_out_it->second.get_node_shared_ptr());
        const auto add = ov::as_type_ptr<ov::op::v1::Add>(add_out_it->second.get_node_shared_ptr());
        const auto fake_quantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_out_it->second.get_node_shared_ptr());
        if (!conv || !mul || !add || !fake_quantize) {
            return false;
        }

        // If there's a Subtract (zero-point dequantization), always apply fallback —
        // int8 ACL convolution executor does not support zero-point yet
        const bool has_subtract = ov::is_type<ov::op::v1::Subtract>(conv->get_input_node_ptr(0));
        if (!has_subtract && fake_quantize->get_output_element_type(0) == conv->get_input_element_type(0)) {
            return false;
        }

        return fallback_lp_conv_to_fp16(conv, mul, add);
    };

    auto matcher = std::make_shared<pattern::Matcher>(conv_mul_add_fq.fake_quantize, "FallbackUnsupportedLPConvToFP16");
    register_matcher(matcher, callback);

    const auto conv_mul_add = create_conv_mul_add_pattern();
    ov::matcher_pass_callback no_fq_callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out_it = pattern_map.find(conv_mul_add.convolution);
        const auto mul_out_it = pattern_map.find(conv_mul_add.multiply);
        const auto add_out_it = pattern_map.find(conv_mul_add.add);
        if (conv_out_it == pattern_map.end() || mul_out_it == pattern_map.end() || add_out_it == pattern_map.end()) {
            return false;
        }

        const auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(conv_out_it->second.get_node_shared_ptr());
        const auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(mul_out_it->second.get_node_shared_ptr());
        const auto add = ov::as_type_ptr<ov::op::v1::Add>(add_out_it->second.get_node_shared_ptr());
        if (!conv || !mul || !add) {
            return false;
        }

        return fallback_lp_conv_to_fp16(conv, mul, add);
    };

    auto no_fq_matcher = std::make_shared<pattern::Matcher>(conv_mul_add.add, "FallbackUnsupportedLPConvToFP16NoFQ");
    register_matcher(no_fq_matcher, no_fq_callback);
}
