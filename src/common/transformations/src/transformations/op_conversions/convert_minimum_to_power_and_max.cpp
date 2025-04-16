// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertMinimum::ConvertMinimum() {
    MATCHER_SCOPE(ConvertMinimum);
    auto minimum = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto minimum = ov::as_type_ptr<ov::op::v1::Minimum>(m.get_match_root());
        if (!minimum || transformation_callback(minimum) || !minimum->get_output_element_type(0).is_signed()) {
            return false;
        }

        /*
         * Decompose Minimum operation to Mul(-1)---->Maximum-->Mul(-1)
         *                                Mul(-1)--'
         */

        auto neg_0 = std::make_shared<ov::op::v1::Multiply>(
            minimum->input(0).get_source_output(),
            ov::op::v0::Constant::create(minimum->get_input_element_type(0), Shape{}, {-1}));

        auto neg_1 = std::make_shared<ov::op::v1::Multiply>(
            minimum->input(1).get_source_output(),
            ov::op::v0::Constant::create(minimum->get_input_element_type(1), Shape{}, {-1}));

        auto max = std::make_shared<ov::op::v1::Maximum>(neg_0, neg_1);

        auto neg_2 = std::make_shared<ov::op::v1::Multiply>(
            max,
            ov::op::v0::Constant::create(max->get_element_type(), Shape{}, {-1}));

        if (minimum->get_input_element_type(0).is_signed() && minimum->get_input_element_type(0).is_integral_number() &&
            minimum->get_input_element_type(1).is_signed() && minimum->get_input_element_type(1).is_integral_number()) {
            const auto min_values = ov::util::make_tensor_of_min_value(max->get_input_element_type(0));
            const auto min_constant = std::make_shared<ov::op::v0::Constant>(min_values);

            const auto is_min_0 = std::make_shared<op::v1::Equal>(minimum->input(0).get_source_output(), min_constant);
            const auto is_min_1 = std::make_shared<op::v1::Equal>(minimum->input(1).get_source_output(), min_constant);
            const auto select_0 = std::make_shared<op::v1::Select>(is_min_0, min_constant, neg_2);
            const auto select_1 = std::make_shared<op::v1::Select>(is_min_1, min_constant, select_0);

            select_1->set_friendly_name(minimum->get_friendly_name());
            ov::copy_runtime_info(minimum, {neg_0, neg_1, max, neg_2, is_min_0, is_min_1, select_0, select_1});
            ov::replace_node(minimum, select_1);
        }

        ov::replace_node(minimum, neg_2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(minimum, matcher_name);
    this->register_matcher(m, callback);
}
