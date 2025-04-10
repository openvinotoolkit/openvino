// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_leaky_relu.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/leaky_relu.hpp"

ov::intel_cpu::ConvertToLeakyRelu::ConvertToLeakyRelu() {
    MATCHER_SCOPE(ConvertToLeakyRelu);
    auto input = ov::pass::pattern::any_input();
    auto slope_constant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    auto prelu = ov::pass::pattern::wrap_type<ov::opset1::PRelu>({input, slope_constant});

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto prelu = ov::as_type_ptr<ov::opset1::PRelu>(m.get_match_root());
        if (!prelu) {
            return false;
        }
        auto slopeNode = ov::as_type_ptr<ov::opset1::Constant>(prelu->get_input_node_shared_ptr(1));
        if (slopeNode != nullptr && ov::shape_size(slopeNode->get_shape()) == 1) {
            const float slope = slopeNode->cast_vector<float>()[0];
            const auto leakyRelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(prelu->input(0).get_source_output(),
                                                                                  slope,
                                                                                  prelu->output(0).get_element_type());
            leakyRelu->set_friendly_name(prelu->get_friendly_name());
            ov::copy_runtime_info(prelu, leakyRelu);
            ov::replace_node(prelu, leakyRelu);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(prelu, matcher_name);
    this->register_matcher(m, callback);
}
