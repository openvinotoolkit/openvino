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

        auto rv_var_id = readvalue->get_variable()->get_info().variable_id;

        // Mark ReadValue corresponding Assign node.
        bool found_assign = false;
        for (size_t i = 0; i < readvalue->get_output_size(); i++) {
            auto son = readvalue->get_output_target_inputs(i).begin()->get_node()->shared_from_this();
            auto assign = as_type_ptr<ov::opset6::Assign>(son);
            if (assign) {
                if (assign->get_variable_id() != rv_var_id) {
                    return false;
                }
                son->set_variable_id(rv_var_id);
                found_assign = true;
                break;
            }
        }
        if (!found_assign) {
            return false;
        }

        // Loop all parent nodes and mark them.
        int recursive_deep = 0;
#define MAX_RECURSIVE_DEEP_MARK_NODE 10
        std::function<void(std::shared_ptr<ov::Node>, std::string)> mark_node =
            [&mark_node, &rv_var_id, &recursive_deep](std::shared_ptr<ov::Node> node, std::string rv_friendly_name) {
                if (op::util::is_parameter(node)) {
                    return;
                }
                if (op::util::is_constant(node)) {
                    return;
                }
                recursive_deep++;
                if (recursive_deep > MAX_RECURSIVE_DEEP_MARK_NODE) {
                    return;
                }

                // Check whether current node have same successor[rv_friendly_name].
                int rd = 0;
#define MAX_RECURSIVE_DEEP_SUCCESSOR 10
                bool final_successor_is_rv = true;
                std::function<void(std::shared_ptr<ov::Node>)> check_successor =
                    [&final_successor_is_rv, &check_successor, &rd, &rv_friendly_name](std::shared_ptr<ov::Node> node) {
                        rd++;
                        if (rd > MAX_RECURSIVE_DEEP_SUCCESSOR) {
                            final_successor_is_rv = false;
                            return;
                        }
                        for (size_t i = 0; i < node->get_output_size(); i++) {
                            auto cur = node->get_output_target_inputs(i).begin()->get_node()->shared_from_this();
                            if (cur->get_friendly_name() != rv_friendly_name) {
                                check_successor(cur);
                            }
                        }
                        rd--;
                    };
                check_successor(node);
                if (final_successor_is_rv) {
                    node->set_state_name(rv_var_id);
                    std::cout << "== Mark Node: " << node->get_friendly_name() << ", set_state_name: " << rv_var_id
                              << ", node tyne=" << node->get_type_name() << std::endl;
                } else {
                    return;
                }

                for (size_t i = 0; i < node->get_input_size(); i++) {
                    mark_node(node->get_input_node_shared_ptr(i), rv_friendly_name);
                }
                recursive_deep--;
            };

        for (size_t i = 0; i < readvalue->get_input_size(); i++) {
            mark_node(readvalue->get_input_node_shared_ptr(i), readvalue->get_friendly_name());
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(readvalue_pattern, "MarkReadValueInputs");
    this->register_matcher(m, callback);
}