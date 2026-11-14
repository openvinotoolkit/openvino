// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fallback_unsupported_lp_conv_to_float.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "conv_mul_add_fq_block.hpp"
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
#include "openvino/pass/pattern/op/pattern.hpp"
#include "ov_ops/type_relaxed.hpp"

using namespace ov::pass;

namespace {
std::pair<ov::Output<ov::Node>, std::shared_ptr<ov::Node>> ensure_precision(
    const ov::Output<ov::Node>& input,
    const ov::element::Type& precision,
    const std::string& convert_name,
    const std::shared_ptr<ov::Node>& rt_source) {
    if (input.get_element_type() == precision) {
        return {input, nullptr};
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(input, precision);
    convert->set_friendly_name(convert_name);
    ov::copy_runtime_info(rt_source, convert);
    return {convert->output(0), convert};
}

}  // namespace

ov::intel_cpu::FallbackUnsupportedLPConvToFloat::FallbackUnsupportedLPConvToFloat() {
    auto conv_mul_add_fq = std::make_shared<ov::intel_cpu::ConvMulAddFQBlock>(false);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv_out = conv_mul_add_fq->get_anchor("gemm", pattern_map);
        const auto mul_out = conv_mul_add_fq->get_anchor("multiply", pattern_map);
        const auto add_out = conv_mul_add_fq->get_anchor("add", pattern_map);
        const auto fq_out = conv_mul_add_fq->get_anchor("fake_quantize", pattern_map);
        if (!conv_out || !mul_out || !add_out || !fq_out) {
            return false;
        }

        const auto conv = ov::as_type_ptr<ov::op::v1::Convolution>(conv_out->get_node_shared_ptr());
        const auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(mul_out->get_node_shared_ptr());
        const auto add = ov::as_type_ptr<ov::op::v1::Add>(add_out->get_node_shared_ptr());
        const auto fake_quantize = ov::as_type_ptr<ov::op::v0::FakeQuantize>(fq_out->get_node_shared_ptr());
        if (!conv || !mul || !add || !fake_quantize) {
            return false;
        }

        // If there's a Subtract (zero-point dequantization), always apply fallback —
        // int8 ACL convolution executor does not support zero-point yet
        const bool has_subtract = ov::is_type<ov::op::v1::Subtract>(conv->get_input_node_ptr(0));
        if (!has_subtract && fake_quantize->get_output_element_type(0) == conv->get_input_element_type(0)) {
            return false;
        }

        // Match the floating Add output so ACL receives consistent source, weights,
        // bias, and destination types. A mixed f16 -> f32 convolution uses reference.
        const auto precision =
            add->get_output_element_type(0) == ov::element::f32 ? ov::element::f32 : ov::element::f16;
        const std::string precision_suffix = precision == ov::element::f32 ? "FP32" : "FP16";
        constexpr size_t scales_port = 1;
        const auto rank_value = conv->get_input_shape(1).size();
        auto [activation_float, activation_convert] =
            ensure_precision(conv->input_value(0),
                             precision,
                             conv->get_friendly_name() + "/ActivationTo" + precision_suffix,
                             conv);

        auto [weights_float, weights_convert] =
            ensure_precision(conv->input_value(1),
                             precision,
                             conv->get_friendly_name() + "/WeightsTo" + precision_suffix,
                             conv);
        auto [scales_float, scales_convert] =
            ensure_precision(mul->input_value(scales_port),
                             precision,
                             mul->get_friendly_name() + "/ScalesTo" + precision_suffix,
                             mul);

        std::vector<int64_t> reshape_pattern(static_cast<size_t>(rank_value), 1);
        reshape_pattern[0] = -1;
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i64, {reshape_pattern.size()}, reshape_pattern);
        auto scales_reshape = std::make_shared<ov::op::v1::Reshape>(scales_float, reshape_const, true);
        scales_reshape->set_friendly_name(mul->get_friendly_name() + "/ScalesToWeightsShape");

        auto scales_to_weights = std::make_shared<ov::op::v1::Multiply>(weights_float, scales_reshape);
        scales_to_weights->set_friendly_name(mul->get_friendly_name() + "/WeightsScaled");
        auto conv_scaled = conv->clone_with_new_inputs({activation_float, scales_to_weights->output(0)});
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

        // Preserve the matched floating output precision after cloning the type-relaxed Convolution.
        if (add->get_output_element_type(0) == precision) {
            if (auto type_relaxed = std::dynamic_pointer_cast<ov::op::TypeRelaxedBase>(conv_scaled)) {
                type_relaxed->set_overridden_output_type(precision, 0);
                conv_scaled->validate_and_infer_types();
            }
        }

        return true;
    };

    auto matcher = std::make_shared<pattern::Matcher>(conv_mul_add_fq, "FallbackUnsupportedLPConvToFloat");
    register_matcher(matcher, callback);
}
