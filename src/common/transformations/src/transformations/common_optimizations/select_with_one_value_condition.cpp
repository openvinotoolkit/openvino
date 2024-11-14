// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/select_with_one_value_condition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::op::util;

ov::pass::SelectWithOneValueCondition::SelectWithOneValueCondition() {
    MATCHER_SCOPE(SelectWithOneValueCondition);

    auto condition = pattern::wrap_type<ov::op::v0::Constant>();
    auto then_branch = pattern::any_input();
    auto else_branch = pattern::any_input();
    auto select_pattern = make_shared<ov::op::v1::Select>(condition, then_branch, else_branch);

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry copy_from;
        NodeRegistry copy_to;
        auto& pattern_map = m.get_pattern_value_map();
        auto& select_value = pattern_map.at(select_pattern);
        auto select = ov::as_type_ptr<ov::op::v1::Select>(select_value.get_node_shared_ptr());
        if (!select) {
            return false;
        }

        auto condition_value = pattern_map.at(condition);
        auto condition_const = ov::as_type_ptr<ov::op::v0::Constant>(condition_value.get_node_shared_ptr());
        if (!condition_const) {
            return false;
        }
        if (condition_value.get_element_type() != element::boolean) {
            return false;
        }

        // check if all elements in the condition to be true or false
        // only in this case, one certain branch can be selected
        auto cond_value = condition_const->get_vector<bool>();
        if (cond_value.size() == 0) {
            return false;
        }
        auto cond_elem = cond_value[0];
        auto all_equal = all_of(cond_value.begin(), cond_value.end(), [=](bool v) {
            return v == cond_elem;
        });
        if (!all_equal) {
            return false;
        }

        // based on the condition value, mark the selected branch and skipped branch index
        auto branch_index = cond_elem ? 1 : 2;

        // based on the resulted shape and the shape of the skipped branch, perform further steps
        auto select_shape = select->get_output_partial_shape(0);
        auto branch_output = select->input_value(branch_index);
        auto branch_output_shape = branch_output.get_partial_shape();

        if (select_shape.is_static() && branch_output_shape.same_scheme(select_shape)) {
            // Broadcast is not needed if the select shape is exactly the same as the selected branch
            return replace_output_update_name(select->output(0), branch_output);
        } else if (select_shape.is_static()) {
            // if the shape of the selected branch is not the same, it needs the broadcasting
            NodeRegistry copy_to;
            auto select_rank = select_shape.size();
            vector<int32_t> select_shape_values(select_rank);
            for (size_t i = 0; i < select_rank; ++i) {
                select_shape_values[i] = static_cast<int32_t>(select_shape[i].get_length());
            }

            auto target_shape =
                copy_to.make<ov::op::v0::Constant>(element::i32, Shape{select_rank}, select_shape_values);
            auto broadcast = copy_to.make<ov::op::v3::Broadcast>(branch_output, target_shape);
            select->output(0).replace(broadcast->output(0));
            broadcast->set_friendly_name(select->get_friendly_name());
            copy_runtime_info(select, copy_to.get());
        } else {
            return false;
        }

        return true;
    };

    auto m = make_shared<pattern::Matcher>(select_pattern, matcher_name);
    this->register_matcher(m, callback);
}
