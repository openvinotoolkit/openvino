// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_swish_cpu.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset4.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/swish_cpu.hpp"

ov::intel_cpu::ConvertToSwishCPU::ConvertToSwishCPU() {
    MATCHER_SCOPE(ConvertToSwishCPU);
    auto swish = ov::pass::pattern::wrap_type<ov::opset4::Swish>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto swish = ov::as_type_ptr<ov::opset4::Swish>(m.get_match_root());
        if (!swish) {
            return false;
        }
        float beta_value = 1.0;
        if (swish->input_values().size() == 2) {
            auto beta = ov::as_type_ptr<ov::opset4::Constant>(swish->get_input_node_shared_ptr(1));

            if (!beta || ov::shape_size(swish->get_input_shape(1)) != 1) {
                return false;
            }
            beta_value = beta->cast_vector<float>()[0];
        }

        auto swish_cpu = std::make_shared<ov::intel_cpu::SwishNode>(swish->input(0).get_source_output(), beta_value);
        swish_cpu->set_friendly_name(swish->get_friendly_name());
        ov::copy_runtime_info(swish, swish_cpu);
        ov::replace_node(swish, swish_cpu);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(swish, matcher_name);
    this->register_matcher(m, callback);
}
