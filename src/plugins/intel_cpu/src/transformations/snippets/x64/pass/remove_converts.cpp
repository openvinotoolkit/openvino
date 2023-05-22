// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remove_converts.hpp"

#include "snippets/itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "snippets/op/convert_saturation.hpp"

ov::intel_cpu::pass::RemoveConverts::RemoveConverts() {
    MATCHER_SCOPE(RemoveConverts);
    auto parent_convert_wrap = ov::pass::pattern::wrap_type<snippets::op::ConvertSaturation>();
    auto child_convert_wrap = ov::pass::pattern::wrap_type<snippets::op::ConvertSaturation>({ parent_convert_wrap });

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "ov::intel_cpu::pass::RemoveConverts")
        const auto& pm = m.get_pattern_value_map();
        const auto parent_convert = pm.at(parent_convert_wrap).get_node_shared_ptr();
        const auto child_convert = pm.at(child_convert_wrap).get_node_shared_ptr();
        if (
            (parent_convert->get_input_element_type(0) != element::f32) ||
            (parent_convert->get_output_target_inputs(0).size() != 1ull) ||
            (parent_convert->get_output_element_type(0) != element::bf16) ||
            (child_convert->get_output_element_type(0) != element::f32)) {
            return false;
        }

        replace_output_update_name(child_convert->output(0), parent_convert->get_input_source_output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(child_convert_wrap, matcher_name);
    register_matcher(m, callback);
}
