// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_converts.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/convert_saturation.hpp"

ov::intel_cpu::pass::RemoveConverts::RemoveConverts() {
    using namespace ov::pass::pattern;
    MATCHER_SCOPE(RemoveConverts);
    auto input_m = any_input(type_matches(ov::element::f32));
    auto parent_convert_m = wrap_type<snippets::op::ConvertSaturation>({input_m}, type_matches(ov::element::bf16));
    auto child_convert_wrap =
        wrap_type<snippets::op::ConvertSaturation>({parent_convert_m}, type_matches(ov::element::f32));

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RemoveConverts")
        const auto& pm = m.get_pattern_value_map();
        const auto parent_convert = pm.at(parent_convert_m).get_node_shared_ptr();
        const auto child_convert = pm.at(child_convert_wrap).get_node_shared_ptr();

        const auto& parent_convert_consumers = parent_convert->get_output_target_inputs(0);
        for (const auto& input : parent_convert_consumers) {
            const auto node = input.get_node();
            if (ov::is_type<snippets::op::ConvertSaturation>(node) &&
                node->get_output_element_type(0) == child_convert->get_output_element_type(0)) {
                replace_output_update_name(node->output(0), parent_convert->input_value(0));
            }
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(child_convert_wrap, matcher_name);
    register_matcher(m, callback);
}
