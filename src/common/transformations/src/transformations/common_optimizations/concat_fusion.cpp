// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/concat_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::ConcatFusion::ConcatFusion() {
    MATCHER_SCOPE(ConcatFusion);
    auto has_same_axis_concat_input = [](const Output<Node>& output) {
        const auto& concat = ov::as_type_ptr<v0::Concat>(output.get_node_shared_ptr());
        const auto axis = concat->get_axis();
        auto is_aplicable = false;
        for (auto input : concat->input_values()) {
            const auto inp_concat = ov::as_type_ptr<v0::Concat>(input.get_node_shared_ptr());
            if (inp_concat && inp_concat->get_axis() == axis && inp_concat->output(0).get_target_inputs().size() == 1) {
                is_aplicable = true;
            }
        }
        return is_aplicable;
    };
    auto concat_pattern = pattern::wrap_type<v0::Concat>(has_same_axis_concat_input);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();

        const auto& concat = ov::as_type_ptr<v0::Concat>(pattern_map.at(concat_pattern));
        const auto axis = concat->get_axis();

        OutputVector new_inputs;
        for (auto input : concat->input_values()) {
            const auto inp_concat = ov::as_type_ptr<v0::Concat>(input.get_node_shared_ptr());
            if (inp_concat && inp_concat->get_axis() == axis && inp_concat->output(0).get_target_inputs().size() == 1) {
                const auto inp_concat_inps = inp_concat->input_values();
                new_inputs.insert(new_inputs.end(), inp_concat_inps.begin(), inp_concat_inps.end());
            } else {
                new_inputs.push_back(input);
            }
        }

        auto new_concat = std::make_shared<v0::Concat>(new_inputs, axis);
        replace_node(concat, new_concat);
        new_concat->set_friendly_name(concat->get_friendly_name());
        copy_runtime_info(concat, new_concat);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(concat_pattern, matcher_name);
    this->register_matcher(m, callback);
}
