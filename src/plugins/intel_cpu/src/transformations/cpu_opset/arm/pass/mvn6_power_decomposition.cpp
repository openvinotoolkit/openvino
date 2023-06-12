// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "mvn6_power_decomposition.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::MVN6PowerDecomposition::MVN6PowerDecomposition() {
    auto constant_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto power_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Power>({
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        constant_pattern});

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(power_pattern, "MVN6PowerDecomposition"), [=] (ngraph::pattern::Matcher& m) {
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

        auto power_value = constant->cast_vector<float>()[0];
        auto base_degree_value = power->input_value(0).get_node_shared_ptr();
        if (power_value == 2) {
            auto mul = std::make_shared<ngraph::opset1::Multiply>(base_degree_value, base_degree_value);
            mul->set_friendly_name(power->get_friendly_name());
            ngraph::copy_runtime_info({power, constant}, mul);
            ngraph::replace_node(power, mul);
            return true;
        }
        return false;
    });
}
