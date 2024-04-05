// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/variadic_split_to_slice.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <string>
#include <transformations/utils/utils.hpp>

#include "split_fc.hpp"

#include "itt.hpp"

ov::intel_cpu::MoveConvertThroughVariadicSplit::MoveConvertThroughVariadicSplit() {
    MATCHER_SCOPE(VariadicSplitToSlice);
    auto weights_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights_m});
    auto axis_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto split_lengths_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto vs_m = ov::pass::pattern::wrap_type<ov::op::v1::VariadicSplit>({convert_m, axis_m, split_lengths_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& convert_before_vs_output = pattern_map.at(convert_m);
        const auto& vs_output = pattern_map.at(vs_m);
        const auto& weights_output = pattern_map.at(weights_m);

        auto vs = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(vs_output.get_node_shared_ptr());
        auto convert_before_vs =
            std::dynamic_pointer_cast<ov::op::v0::Convert>(convert_before_vs_output.get_node_shared_ptr());

        auto new_vs = vs->clone_with_new_inputs({weights_output, vs->input_value(1), vs->input_value(2)});
        // NodeVector converts_after_vs;
        for (size_t i = 0; i < vs->outputs().size(); i++) {
            auto convert =
                std::make_shared<ov::op::v0::Convert>(new_vs->output(i), convert_before_vs->get_convert_element_type());
            // converts_after_vs.push_back(convert);

            for (auto& target_input : vs->output(i).get_target_inputs()) {
                target_input.replace_source_output(convert);
            }
            convert->set_friendly_name(vs->get_friendly_name() + "." + std::to_string(i));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(vs_m, matcher_name);
    this->register_matcher(m, callback);
}
