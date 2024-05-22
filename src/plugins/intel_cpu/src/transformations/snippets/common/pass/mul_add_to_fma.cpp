// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "mul_add_to_fma.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/snippets/common/op/fused_mul_add.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/op/memory_access.hpp"

ov::intel_cpu::pass::MulAddToFMA::MulAddToFMA() {
    MATCHER_SCOPE(MulAddToFMA);
    auto is_not_memory_access = [](const Output<Node>& out) {
        return !std::dynamic_pointer_cast<const snippets::modifier::MemoryAccess>(out.get_node_shared_ptr());
    };
    auto mul_input_1 = ov::pass::pattern::any_input();
    auto mul_input_2 = ov::pass::pattern::any_input();
    auto mul_m = ov::pass::pattern::wrap_type<opset1::Multiply>({ mul_input_1, mul_input_2 },
                                                                [=](const Output<Node>& out) {
                                                                    return out.get_target_inputs().size() == 1 &&
                                                                           is_not_memory_access(out);
                                                                });
    auto add_input_2 = ov::pass::pattern::any_input();
    auto add_m = ov::pass::pattern::wrap_type<opset1::Add>({ mul_m, add_input_2 }, is_not_memory_access);

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::MulAddToFMA_callback")
        const auto& pattern_map = m.get_pattern_value_map();
        const auto multiply = pattern_map.at(mul_m).get_node_shared_ptr();
        const auto add = pattern_map.at(add_m).get_node_shared_ptr();

        if (transformation_callback(add)) {
            return false;
        }

        const auto& a = multiply->input_value(0);
        const auto& b = multiply->input_value(1);
        const auto& c = pattern_map.at(add_input_2);

        const auto fma = std::make_shared<ov::intel_cpu::FusedMulAdd>(a, b, c);
        ov::copy_runtime_info({ a.get_node_shared_ptr(), b.get_node_shared_ptr(), c.get_node_shared_ptr() }, fma);
        fma->set_friendly_name(add->get_friendly_name());
        ov::replace_node(add, fma);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "MulAddToFMA");
    register_matcher(m, callback);
}
