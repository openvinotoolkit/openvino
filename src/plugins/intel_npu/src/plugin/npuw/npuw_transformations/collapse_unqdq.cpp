// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "collapse_unqdq.hpp"

#include <memory>
#include <string>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
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

class CollapseUNQDQChain final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::CollapseUNQDQChain");

    CollapseUNQDQChain() {
        auto multiply_pattern = opp::wrap_type<ov::op::v1::Multiply>();

        ov::matcher_pass_callback callback = [](opp::Matcher& matcher) {
            auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(matcher.get_match_root());
            if (multiply == nullptr) {
                return false;
            }

            std::shared_ptr<ov::op::v1::Subtract> subtract;
            if (auto candidate =
                    ov::as_type_ptr<ov::op::v1::Subtract>(multiply->input_value(0).get_node_shared_ptr())) {
                subtract = candidate;
            } else if (auto candidate =
                           ov::as_type_ptr<ov::op::v1::Subtract>(multiply->input_value(1).get_node_shared_ptr())) {
                subtract = candidate;
            } else {
                return false;
            }

            const auto dequantized_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(subtract->input_value(0).get_node_shared_ptr());
            if (dequantized_convert == nullptr) {
                return false;
            }

            const auto quantized_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(dequantized_convert->input_value(0).get_node_shared_ptr());
            if (quantized_convert == nullptr) {
                return false;
            }

            const auto fake_quantize =
                ov::as_type_ptr<ov::op::v0::FakeQuantize>(quantized_convert->input_value(0).get_node_shared_ptr());
            if (fake_quantize == nullptr) {
                return false;
            }

            ov::Output<ov::Node> replacement = fake_quantize->input_value(0);
            replacement =
                preserve_output_type(replacement, multiply->get_output_element_type(0), multiply->get_friendly_name());

            ov::replace_node(multiply, ov::OutputVector{replacement});
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(multiply_pattern, "CollapseUNQDQChain"), callback);
    }
};

}  // namespace

ov::npuw::CollapseUNQDQ::CollapseUNQDQ() {
    add_matcher<CollapseUNQDQChain>();
}
