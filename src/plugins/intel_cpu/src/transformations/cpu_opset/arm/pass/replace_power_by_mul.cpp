// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "replace_power_by_mul.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::ReplacePowerByMul::ReplacePowerByMul() {
    auto constant_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto power_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Power>({
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        constant_pattern});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(power_pattern, "ReplacePowerByMul"), [=] (ngraph::pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto power = std::dynamic_pointer_cast<ngraph::opset1::Power>(pattern_map[power_pattern].get_node_shared_ptr());
        if (!power) {
            return false;
        }
        auto constant = std::dynamic_pointer_cast<ngraph::opset1::Constant>(pattern_map[constant_pattern].get_node_shared_ptr());
        if (!constant) {
            return false;
        }
        if (ov::shape_size(constant->get_shape()) != 1) {
            return false;
        }
        auto p = constant->cast_vector<float>()[0];

        if (p == 1) {
            return ov::replace_output_update_name(power->output(0), power->input_value(0));
        } else if (p == 2) {
            auto mul = std::make_shared<ngraph::opset1::Multiply>(
                power->input_value(0).get_node_shared_ptr(),
                power->input_value(0).get_node_shared_ptr());
            mul->set_friendly_name(power->get_friendly_name());
            ngraph::copy_runtime_info({power, constant}, mul);
            ngraph::replace_node(power, mul);
            return true;
        }
        return false;
    });
}
