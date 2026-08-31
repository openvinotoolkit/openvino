// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_gated_residual.hpp"

#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForGatedResidualPattern::DisableFP16CompForGatedResidualPattern() {
    using namespace ov::pass::pattern;

    auto mvn_m = wrap_type<ov::op::v6::MVN>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto mvn = ov::as_type_ptr<ov::op::v6::MVN>(m.get_match_root());
        if (!mvn || mvn->get_output_element_type(0) != element::f32 || transformation_callback(mvn))
            return false;

        auto residual_add = ov::as_type_ptr<ov::op::v1::Add>(mvn->get_input_node_shared_ptr(0));
        if (!residual_add)
            return false;

        bool multiply_found = false;
        for (const auto& input : residual_add->input_values()) {
            const auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(input.get_node_shared_ptr());
            if (!multiply)
                continue;

            multiply_found = true;
            for (const auto& multiply_input : multiply->input_values()) {
                const auto producer = multiply_input.get_node_shared_ptr();
                ov::disable_conversion(producer, element::f16);

                const auto linear_add = ov::as_type_ptr<ov::op::v1::Add>(producer);
                if (!linear_add)
                    continue;

                for (const auto& linear_input : linear_add->input_values()) {
                    const auto linear_producer = linear_input.get_node_shared_ptr();
                    if (ov::is_type<ov::op::v0::MatMul>(linear_producer))
                        ov::disable_conversion(linear_producer, element::f16);
                }
            }

            ov::disable_conversion(multiply, element::f16);
        }
        if (!multiply_found)
            return false;

        ov::disable_conversion(residual_add, element::f16);
        ov::disable_conversion(mvn, element::f16);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mvn_m, "DisableFP16CompForGatedResidualPattern");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
