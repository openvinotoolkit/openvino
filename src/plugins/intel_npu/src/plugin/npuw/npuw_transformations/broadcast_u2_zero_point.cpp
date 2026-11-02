// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_u2_zero_point.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

constexpr float MAX_U2_VALUE = 3.f;

}  // namespace

class BroadcastU2ZeroPointMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::BroadcastU2ZeroPointMatcher");

    BroadcastU2ZeroPointMatcher() {
        auto weights = opp::wrap_type<ov::op::v0::Constant>();
        auto weights_convert = opp::wrap_type<ov::op::v0::Convert>({weights});
        auto zero_point = opp::wrap_type<ov::op::v0::Constant>();
        auto subtract = opp::wrap_type<ov::op::v1::Subtract>({weights_convert, zero_point});

        ov::matcher_pass_callback callback = [=](opp::Matcher& matcher) {
            const auto& pattern_map = matcher.get_pattern_value_map();
            const auto weights_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights).get_node_shared_ptr());
            const auto zero_point_const =
                ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(zero_point).get_node_shared_ptr());
            if (weights_const->get_element_type() != ov::element::u2 || weights_const->get_shape().size() < 2 ||
                !zero_point_const->get_element_type().is_real() || !zero_point_const->get_shape().empty()) {
                return false;
            }
            const float scalar = zero_point_const->cast_vector<float>().front();
            if (scalar != std::floor(scalar) || scalar < 0.f || scalar > MAX_U2_VALUE) {
                return false;
            }

            ov::Shape group_shape = weights_const->get_shape();
            group_shape.back() = 1;
            auto broadcast = ov::op::v0::Constant::create(
                ov::element::u2,
                group_shape,
                std::vector<uint8_t>(ov::shape_size(group_shape), static_cast<uint8_t>(scalar)));
            broadcast->set_friendly_name(zero_point_const->get_friendly_name());
            const auto subtract_node = pattern_map.at(subtract).get_node_shared_ptr();
            subtract_node->input(1).replace_source_output(
                std::make_shared<ov::op::v0::Convert>(broadcast, subtract_node->get_input_element_type(1)));
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(subtract, "BroadcastU2ZeroPointMatcher"), callback);
    }
};

ov::npuw::BroadcastU2ZeroPoint::BroadcastU2ZeroPoint() {
    add_matcher<BroadcastU2ZeroPointMatcher>();
}
