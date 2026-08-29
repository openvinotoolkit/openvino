// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "drop_zp_subtract.hpp"

#include <algorithm>
#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

bool is_all_zero_constant(const ov::Output<ov::Node>& output) {
    const auto constant = ov::util::get_constant_from_source(output);
    if (constant == nullptr) {
        return false;
    }

    const auto values = constant->cast_vector<double>();
    return std::all_of(values.begin(), values.end(), [](double value) {
        return value == 0.0;
    });
}

}  // namespace

class DropZeroPointSubtractMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::DropZeroPointSubtractMatcher");

    DropZeroPointSubtractMatcher() {
        auto subtract_pattern = opp::wrap_type<ov::op::v1::Subtract>();

        ov::matcher_pass_callback callback = [](opp::Matcher& matcher) {
            const auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(matcher.get_match_root());
            if (subtract == nullptr) {
                return false;
            }

            if (!is_all_zero_constant(subtract->input_value(1))) {
                return false;
            }

            ov::replace_node(subtract, ov::OutputVector{subtract->input_value(0)});
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(subtract_pattern, "DropZeroPointSubtractMatcher"), callback);
    }
};

ov::npuw::DropZPSubtract::DropZPSubtract() {
    add_matcher<DropZeroPointSubtractMatcher>();
}
