// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_readvalue_inputs.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MarkReadValueInputs::MarkReadValueInputs() {
    MATCHER_SCOPE(MarkReadValueInputs);
    auto readvalue_pattern = pass::pattern::wrap_type<ov::op::v6::ReadValue>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto readvalue = as_type_ptr<ov::opset6::ReadValue>(pattern_map.at(readvalue_pattern).get_node_shared_ptr());
        if (!readvalue) {
            return false;
        }

        auto state_name = readvalue->get_variable()->get_info().variable_id;
        // Mark ReadValue corresponding Assign node.
        int assign_num = 0;
        int assign_id = 0;
        for (size_t i = 0; i < readvalue->get_output_size(); i++) {
            auto son = readvalue->get_output_target_inputs(i).begin()->get_node()->shared_from_this();
            if (as_type_ptr<ov::opset6::Assign>(son)) {
                assign_num++;
                assign_id = i;
            }
        }
        if (assign_num == 1) {
            readvalue->get_output_target_inputs(assign_id).begin()->get_node()->shared_from_this()->set_state_name(
                state_name);
        } else {
            return false;
        }

        // Loop all parent nodes and mark them.
        int recursive_deep = 0;
        auto cur = readvalue;

        // Mark parent nodes.
        recursive_deep = 0;
        std::function<void(std::shared_ptr<ov::Node>)> mark_node = [&mark_node, &state_name, &recursive_deep](
                                                                       std::shared_ptr<ov::Node> node) {
            if (recursive_deep > 10) {
                return;
            }
            if (op::util::is_parameter(node)) {
                return;
            }
            if (op::util::is_constant(node)) {
                return;
            }

            if (node->outputs().size() == 1u) {
                if (!as_type_ptr<ov::opset6::ReadValue>(node)) {
                    node->set_state_name(state_name);
                    std::cout << "== Mark Node: " << node->get_friendly_name() << ", set_state_name: " << state_name
                              << ", node tyne=" << node->get_type_name() << std::endl;
                }

                for (size_t i = 0; i < node->get_input_size(); i++) {
                    mark_node(node->get_input_node_shared_ptr(i));
                }
                return;
            }
            recursive_deep++;
        };
        for (size_t i = 0; i < cur->get_input_size(); i++) {
            mark_node(cur->get_input_node_shared_ptr(i));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(readvalue_pattern, "MarkReadValueInputs");
    this->register_matcher(m, callback);
}