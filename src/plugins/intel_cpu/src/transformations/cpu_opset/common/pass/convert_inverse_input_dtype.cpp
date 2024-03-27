// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_inverse_input_dtype.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::intel_cpu::ConvertInverseInputDtype::ConvertInverseInputDtype() {
    MATCHER_SCOPE(ConvertInverseInputDtype);
    auto inverse = ov::pass::pattern::wrap_type<ov::op::v14::Inverse>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto inverse = std::dynamic_pointer_cast<ov::op::v14::Inverse>(m.get_match_root());
        if (!inverse) {
            return false;
        }

        auto data_node = inverse->input_value(0);
        const auto input_element_type = data_node.get_element_type();

        if (input_element_type == ov::element::f32) {
            return false;
        }

        // No support for integer types for Inverse and f64 is not supported in CPU plugin
        if (input_element_type.is_integral() || input_element_type == ov::element::f64) {
            return false;
        }

        ov::NodeVector new_ops;
        auto convert_data = std::make_shared<ov::op::v0::Convert>(data_node, ov::element::f32);
        auto inverse_f32 = std::make_shared<ov::op::v14::Inverse>(convert_data);
        auto convert_output = std::make_shared<ov::op::v0::Convert>(inverse_f32, data_node.get_element_type());

        new_ops.push_back(convert_data);
        new_ops.push_back(inverse_f32);
        new_ops.push_back(convert_output);

        inverse_f32->set_friendly_name(inverse->get_friendly_name());

        ov::copy_runtime_info(inverse, new_ops);
        ov::replace_node(inverse, convert_output);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(inverse, matcher_name);
    this->register_matcher(m, callback);
}
