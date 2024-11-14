// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_minimum_to_power_and_max.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
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

        neg_2->set_friendly_name(minimum->get_friendly_name());
        ov::copy_runtime_info(minimum, {neg_0, neg_1, max, neg_2});
        ov::replace_node(minimum, neg_2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(minimum, matcher_name);
    this->register_matcher(m, callback);
}
