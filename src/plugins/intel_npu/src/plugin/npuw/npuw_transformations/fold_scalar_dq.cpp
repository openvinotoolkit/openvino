// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fold_scalar_dq.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

class FoldScalarDQMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::FoldScalarDQMatcher");

    FoldScalarDQMatcher() {
        auto weight = opp::wrap_type<ov::op::v0::Constant>();
        auto weight_convert = opp::wrap_type<ov::op::v0::Convert>({weight});
        auto zero_point = opp::wrap_type<ov::op::v0::Constant>();
        auto zero_point_convert = opp::optional<ov::op::v0::Convert>({zero_point});
        auto weight_minus_zp = opp::wrap_type<ov::op::v1::Subtract>({weight_convert, zero_point_convert});
        auto zp_minus_weight = opp::wrap_type<ov::op::v1::Subtract>({zero_point_convert, weight_convert});
        auto multiply_input_or =
            std::make_shared<opp::op::Or>(ov::OutputVector{weight_convert, weight_minus_zp, zp_minus_weight});
        auto scale = opp::wrap_type<ov::op::v0::Constant>();
        auto scale_convert = opp::optional<ov::op::v0::Convert>({scale});
        auto multiply = opp::wrap_type<ov::op::v1::Multiply>({multiply_input_or, scale_convert});

        ov::matcher_pass_callback callback = [=](opp::Matcher& matcher) {
            const auto multiply_node = matcher.get_pattern_value_map().at(multiply).get_node_shared_ptr();

            const auto& shape = multiply_node->output(0).get_partial_shape();
            if (shape.is_dynamic() || ov::shape_size(shape.to_shape()) != 1) {
                return false;
            }

            const auto folded = ov::util::constantfold_subgraph(multiply_node->output(0));
            if (folded == nullptr) {
                return false;
            }

            folded->set_friendly_name(multiply_node->get_friendly_name());
            ov::copy_runtime_info(matcher.get_matched_nodes(), folded);
            ov::replace_node(multiply_node, folded);
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(multiply, "FoldScalarDQMatcher"), callback);
    }
};

ov::npuw::FoldScalarDQ::FoldScalarDQ() {
    add_matcher<FoldScalarDQMatcher>();
}
