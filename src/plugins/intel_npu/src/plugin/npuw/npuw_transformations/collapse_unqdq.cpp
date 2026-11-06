// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <string>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace opp = ov::pass::pattern;

namespace {

ov::Output<ov::Node> preserve_output_type(const ov::Output<ov::Node>& replacement,
                                          const ov::element::Type& expected_type,
                                          const std::string& friendly_name) {
    if (replacement.get_element_type() == expected_type) {
        return replacement;
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(replacement, expected_type);
    convert->set_friendly_name(friendly_name);
    return convert;
}

// With the zero-point Subtract absent (zp == 0), FQ -> Convert -> Convert -> Multiply(scale)
// is an identity on the FQ input only if the FQ is the matching quantizer:
//   input_low == output_low * scale  and  input_high == output_high * scale.
bool is_identity_quantizer_without_zp(const ov::Output<ov::Node>& in_low_n,
                                      const ov::Output<ov::Node>& in_high_n,
                                      const ov::Output<ov::Node>& out_low_n,
                                      const ov::Output<ov::Node>& out_high_n,
                                      const ov::Output<ov::Node>& scale_n) {
    auto get_constant = [](const ov::Output<ov::Node>& output) -> std::optional<float> {
        const auto constant = ov::util::get_constant_from_source(output);
        if (constant == nullptr) {
            return std::nullopt;
        }
        float value = 0.f;
        return ov::op::util::get_single_value(constant, value) ? std::optional<float>(value) : std::nullopt;
    };
    const auto in_low = get_constant(in_low_n);
    const auto in_high = get_constant(in_high_n);
    const auto out_low = get_constant(out_low_n);
    const auto out_high = get_constant(out_high_n);
    const auto s = get_constant(scale_n);
    if (!in_low || !in_high || !out_low || !out_high || !s) {
        return false;
    }

    // Tolerance: one quantization step (|scale|)
    const float tol = std::abs(*s);
    const auto scaled_out_low = *out_low * *s;
    const auto scaled_out_high = *out_high * *s;
    auto isclose = [](float a, float b, float tolerance) {
        return std::abs(a - b) <= tolerance;
    };
    return isclose(*in_low, scaled_out_low, tol) && isclose(*in_high, scaled_out_high, tol);
}

}  // namespace

class CollapseUNQDQChain final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CollapseUNQDQChain");

    CollapseUNQDQChain() {
        // Pattern:
        //   FakeQuantize(input, ...)
        //     → Convert (quantize, e.g. to u16)
        //       → Convert (dequantize, back to f32)
        //         → [Subtract (zero-point)]   (optional: dropped for zero ZP)
        //           → Multiply (scale)   ← anchor
        auto input = opp::any_input();
        auto fq_in_low = opp::any_input();
        auto fq_in_high = opp::any_input();
        auto fq_out_low = opp::any_input();
        auto fq_out_high = opp::any_input();
        auto fake_q = opp::wrap_type<ov::op::v0::FakeQuantize>({input, fq_in_low, fq_in_high, fq_out_low, fq_out_high});
        auto cvt_q = opp::wrap_type<ov::op::v0::Convert>({fake_q});
        auto cvt_dq = opp::wrap_type<ov::op::v0::Convert>({cvt_q});
        auto subtract = opp::optional<ov::op::v1::Subtract>({cvt_dq, opp::any_input()});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({subtract, opp::any_input()});

        // Note: use [=] so pattern nodes stay alive inside the callback.
        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            const auto matched_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
            const auto matched_input = node_to_output.at(input);

            const bool has_subtract = ov::is_type<ov::op::v1::Subtract>(matched_multiply->get_input_node_ptr(0));
            if (!has_subtract && !is_identity_quantizer_without_zp(node_to_output.at(fq_in_low),
                                                                   node_to_output.at(fq_in_high),
                                                                   node_to_output.at(fq_out_low),
                                                                   node_to_output.at(fq_out_high),
                                                                   matched_multiply->input_value(1))) {
                return false;
            }

            ov::Output<ov::Node> replacement = matched_input;
            replacement = preserve_output_type(replacement,
                                               matched_multiply->get_output_element_type(0),
                                               matched_multiply->get_friendly_name());
            ov::replace_node(matched_multiply, ov::OutputVector{replacement});
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(multiply, "CollapseUNQDQChain"), callback);
    }
};

ov::npuw::CollapseUNQDQ::CollapseUNQDQ() {
    add_matcher<CollapseUNQDQChain>();
}
