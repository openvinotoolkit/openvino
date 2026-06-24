// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <memory>
#include <string>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

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

}  // namespace

class CollapseUNQDQChain final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CollapseUNQDQChain");

    CollapseUNQDQChain() {
        // Pattern:
        //   FakeQuantize(input, ...)
        //     → Convert (quantize, e.g. to u16)
        //       → Convert (dequantize, back to f32)
        //         → Subtract (zero-point)
        //           → Multiply (scale)   ← anchor
        auto input = opp::any_input();
        auto fake_q = opp::wrap_type<ov::op::v0::FakeQuantize>(
            {input, opp::any_input(), opp::any_input(), opp::any_input(), opp::any_input()});
        auto cvt_q = opp::wrap_type<ov::op::v0::Convert>({fake_q});
        auto cvt_dq = opp::wrap_type<ov::op::v0::Convert>({cvt_q});
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({cvt_dq, opp::any_input()});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({subtract, opp::any_input()});

        // Note: use [=] so pattern nodes stay alive inside the callback.
        ov::matcher_pass_callback callback = [=](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            const auto matched_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(m.get_match_root());
            const auto matched_input = node_to_output.at(input);

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
