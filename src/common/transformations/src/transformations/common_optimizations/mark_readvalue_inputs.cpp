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

        // Loop all parent nodes and mark them.
        int recursive_deep = 0;
        auto cur = readvalue;
        auto state_name = readvalue->get_variable()->get_info().variable_id;

        // auto contain_concat = false;
        // std::function<void(std::shared_ptr<ov::Node>)> check_contain_concat =
        //     [&check_contain_concat, &contain_concat](std::shared_ptr<ov::Node> node) {
        //         if (op::util::is_parameter(node)) {
        //             return;
        //         }
        //         if (op::util::is_constant(node)) {
        //             return;
        //         }
        //         if (as_type_ptr<ov::opset1::Concat>(node)) {
        //             contain_concat = true;
        //             return;
        //         }
        //         if (node->get_output_size() == 1) {
        //             for (size_t i = 0; i < node->get_input_size(); i++) {
        //                 check_contain_concat(node->get_input_node_shared_ptr(i));
        //             }
        //             return;
        //         }
        //         return;
        //     };
        // check_contain_concat(cur);
        // if (contain_concat) {
        //     return false;
        // }

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

            if (node->get_output_size() == 1) {
                if (!as_type_ptr<ov::opset6::ReadValue>(node)) {
                    node->set_state_name(state_name);
                    std::cout << "== Mark Node: " << node->get_friendly_name() << ", set_state_name: " << state_name
                              << std::endl;
                }

                for (size_t i = 0; i < node->get_input_size(); i++) {
                    mark_node(node->get_input_node_shared_ptr(i));
                }
                return;
            }
            recursive_deep++;
            return;
        };
        mark_node(cur);

        // Mark assign in the follow 2 layers son nodes
        recursive_deep = 0;
        bool found = false;
        std::function<void(std::shared_ptr<ov::Node>)> mark_assign =
            [&mark_assign, &state_name, &recursive_deep, &found](std::shared_ptr<ov::Node> node) {
                if (recursive_deep > 1) {
                    return;
                }
                auto assign = as_type_ptr<ov::opset6::Assign>(node);
                if (assign) {
                    node->set_state_name(state_name);
                    found = true;
                    return;
                }

                for (size_t i = 0; i < node->get_output_size(); i++) {
                    mark_assign(node->get_output_target_inputs(i).begin()->get_node()->shared_from_this());
                    if (found) {
                        return;
                    }
                }
                recursive_deep++;
            };
        mark_assign(cur);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(readvalue_pattern, "MarkReadValueInputs");
    this->register_matcher(m, callback);
}