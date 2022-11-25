// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/select_with_one_value_condition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset10;
using namespace ov::op::util;

ov::pass::SelectWithOneValueCondition::SelectWithOneValueCondition() {
    MATCHER_SCOPE(SelectWithOneValueCondition);

    auto condition = pattern::wrap_type<Constant>();
    auto then_branch = pattern::any_input();
    auto else_branch = pattern::any_input();
    auto select_pattern = make_shared<Select>(condition, then_branch, else_branch);

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        NodeRegistry copy_from;
        NodeRegistry copy_to;
        auto& pattern_map = m.get_pattern_value_map();
        auto select_value = pattern_map.at(select_pattern);
        auto select = std::dynamic_pointer_cast<Select>(select_value.get_node_shared_ptr());
        if (!select) {
            return false;
        }

        auto condition_value = pattern_map.at(condition);
        auto condition_const = std::dynamic_pointer_cast<Constant>(condition_value.get_node_shared_ptr());
        if (!condition_const) {
            return false;
        }
        if (condition_value.get_element_type() != element::boolean) {
            return false;
        }

        // check if all elements in the condition to be true or false
        auto cond_value = condition_const->get_vector<bool>();
        if (cond_value.size() == 0) {
            return false;
        }
        auto all_true = std::all_of(cond_value.begin(), cond_value.end(), [](bool v) {
            return v;
        });
        auto all_false = std::all_of(cond_value.begin(), cond_value.end(), [](bool v) {
            return !v;
        });
        if (!all_true && !all_false) {
            return false;
        }

        auto cond_elem = cond_value[0];

        if (cond_elem) {
            select->output(0).replace(select->input_value(1));
        } else {
            select->output(0).replace(select->input_value(2));
        }

        return true;
    };

    auto m = make_shared<pattern::Matcher>(select_pattern, matcher_name);
    this->register_matcher(m, callback);
}
