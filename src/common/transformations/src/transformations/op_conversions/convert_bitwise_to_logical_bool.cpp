// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_bitwise_to_logical_bool.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertBitwiseAndToLogicalAnd::ConvertBitwiseAndToLogicalAnd() {
    MATCHER_SCOPE(ConvertBitwiseAndToLogicalAnd);
    auto pattern =
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseAnd>({ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean)),
                                                     ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto bitwise = ov::as_type_ptr<ov::op::v13::BitwiseAnd>(m.get_match_root());
        if (!bitwise || transformation_callback(bitwise)) {
            return false;
        }

        const auto logical = std::make_shared<ov::op::v1::LogicalAnd>(bitwise->input_value(0),
                                                                      bitwise->input_value(1),
                                                                      bitwise->get_autob());

        logical->set_friendly_name(bitwise->get_friendly_name());
        copy_runtime_info(bitwise, logical);
        replace_node(bitwise, logical);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}
ov::pass::ConvertBitwiseNotToLogicalNot::ConvertBitwiseNotToLogicalNot() {
    MATCHER_SCOPE(ConvertBitwiseNotToLogicalNot);
    auto pattern =
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseNot>({ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto bitwise = ov::as_type_ptr<ov::op::v13::BitwiseNot>(m.get_match_root());
        if (!bitwise || transformation_callback(bitwise)) {
            return false;
        }

        const auto logical = std::make_shared<ov::op::v1::LogicalNot>(bitwise->input_value(0));

        logical->set_friendly_name(bitwise->get_friendly_name());
        copy_runtime_info(bitwise, logical);
        replace_node(bitwise, logical);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertBitwiseOrToLogicalOr::ConvertBitwiseOrToLogicalOr() {
    MATCHER_SCOPE(ConvertBitwiseOrToLogicalOr);
    auto pattern =
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseOr>({ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean)),
                                                    ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto bitwise = ov::as_type_ptr<ov::op::v13::BitwiseOr>(m.get_match_root());
        if (!bitwise || transformation_callback(bitwise)) {
            return false;
        }

        const auto logical = std::make_shared<ov::op::v1::LogicalOr>(bitwise->input_value(0),
                                                                     bitwise->input_value(1),
                                                                     bitwise->get_autob());

        logical->set_friendly_name(bitwise->get_friendly_name());
        copy_runtime_info(bitwise, logical);
        replace_node(bitwise, logical);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertBitwiseXorToLogicalXor::ConvertBitwiseXorToLogicalXor() {
    MATCHER_SCOPE(ConvertBitwiseXorToLogicalXor);
    auto pattern =
        ov::pass::pattern::wrap_type<ov::op::v13::BitwiseXor>({ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean)),
                                                     ov::pass::pattern::any_input(ov::pass::pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto bitwise = ov::as_type_ptr<ov::op::v13::BitwiseXor>(m.get_match_root());
        if (!bitwise || transformation_callback(bitwise)) {
            return false;
        }

        const auto logical = std::make_shared<ov::op::v1::LogicalXor>(bitwise->input_value(0),
                                                                      bitwise->input_value(1),
                                                                      bitwise->get_autob());

        logical->set_friendly_name(bitwise->get_friendly_name());
        copy_runtime_info(bitwise, logical);
        replace_node(bitwise, logical);

        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}
