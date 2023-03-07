// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_fuse.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/transpose_sinking_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace opset10;

ov::pass::TransposeSinkingFuse::TransposeSinkingFuse() {
    MATCHER_SCOPE(TransposeFuse);
    auto transpose_label = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::wrap_type<Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto transpose_1 = pattern_to_output.at(transpose_label);
        auto order_const_1 = std::dynamic_pointer_cast<Constant>(transpose_1->input_value(1).get_node_shared_ptr());
        auto consumers = transpose_1->get_output_target_inputs(0);

        std::vector<int64_t> saved_order_values;
        auto saved_type = order_const_1->get_element_type();
        for (const auto& it : consumers) {
            auto out_transpose = dynamic_cast<Transpose*>(it.get_node());
            if (!out_transpose) {
                return false;
            }

            auto order = out_transpose->input_value(1).get_node_shared_ptr();
            auto order_const = std::dynamic_pointer_cast<Constant>(order);
            if (!order_const) {
                return false;
            }

            auto order_values = order_const->cast_vector<int64_t>();
            if (order_values.empty()) {
                return false;
            }

            if (saved_order_values.empty()) {
                saved_order_values = order_values;
            } else {
                if (saved_order_values != order_values) {
                    return false;
                }
            }

            if (order_const->get_element_type() != saved_type) {
                saved_type = element::i64;
            }
        }

        auto order1 = order_const_1->cast_vector<int64_t>();
        if (order1.size() != saved_order_values.size()) {
            return false;
        }

        bool is_ordered = true;
        for (size_t i = 0; i < order1.size(); i++) {
            saved_order_values[i] = order1[saved_order_values[i]];
            if (saved_order_values[i] != (int64_t)i)
                is_ordered = false;
        }

        if (is_ordered) {
            for (const auto& it : consumers) {
                it.get_node()->output(0).replace(transpose_1->input_value(0));
            }
        } else {
            auto new_order = Constant::create(saved_type, {saved_order_values.size()}, saved_order_values);
            auto new_transpose = register_new_node<Transpose>(transpose_1->input_value(0), new_order);
            for (const auto& it : consumers) {
                new_transpose->set_friendly_name(it.get_node()->get_friendly_name());
                it.get_node()->output(0).replace(new_transpose);
                copy_runtime_info(transpose_1, new_transpose);
            }
            transpose_sinking::UpdateForwardSinkingAbility(new_transpose);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
