// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_bitwise_to_logical_bool.hpp"

#include "itt.hpp"
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
        pattern::wrap_type<ov::op::v13::BitwiseAnd>({pattern::any_input(pattern::type_matches(element::boolean)),
                                                     pattern::any_input(pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
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
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}
ov::pass::ConvertBitwiseNotToLogicalNot::ConvertBitwiseNotToLogicalNot() {
    MATCHER_SCOPE(ConvertBitwiseNotToLogicalNot);
    auto pattern =
        pattern::wrap_type<ov::op::v13::BitwiseNot>({pattern::any_input(pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
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
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertBitwiseOrToLogicalOr::ConvertBitwiseOrToLogicalOr() {
    MATCHER_SCOPE(ConvertBitwiseOrToLogicalOr);
    auto pattern =
        pattern::wrap_type<ov::op::v13::BitwiseOr>({pattern::any_input(pattern::type_matches(element::boolean)),
                                                    pattern::any_input(pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
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
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::ConvertBitwiseXorToLogicalXor::ConvertBitwiseXorToLogicalXor() {
    MATCHER_SCOPE(ConvertBitwiseXorToLogicalXor);
    auto pattern =
        pattern::wrap_type<ov::op::v13::BitwiseXor>({pattern::any_input(pattern::type_matches(element::boolean)),
                                                     pattern::any_input(pattern::type_matches(element::boolean))});

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
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
    auto m = std::make_shared<pattern::Matcher>(pattern, matcher_name);
    register_matcher(m, callback);
}
