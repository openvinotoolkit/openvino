// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "itt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::RemoveMultiSubGraphOpDanglingParams::RemoveMultiSubGraphOpDanglingParams() {
    MATCHER_SCOPE(RemoveMultiSubGraphOpDanglingParams);
    auto multi_subgraph_op_pattern = pattern::wrap_type<op::util::MultiSubGraphOp>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(m.get_match_root());
        if (multi_subgraph_op == nullptr) {
            return false;
        }
        bool pass_required = false;
        std::set<ov::Output<ov::Node>> required_inputs;
        auto op_inputs = multi_subgraph_op->input_values();
        std::vector<std::vector<size_t>> to_remove_descriptors_indexes;
        const auto subgraphs_size = multi_subgraph_op->get_internal_subgraphs_size();
        to_remove_descriptors_indexes.resize(subgraphs_size);
        for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
            auto& body_func = multi_subgraph_op->get_function(static_cast<int>(body_idx));
            auto& body_params = body_func->get_parameters();
            auto& body_in_descriptors = multi_subgraph_op->get_input_descriptions(static_cast<int>(body_idx));
            // collect all descriptors which should be removed and reqired inputs
            for (size_t i = 0; i < body_in_descriptors.size(); ++i) {
                auto& body_param = body_params[body_in_descriptors[i]->m_body_parameter_index];
                if (body_param->get_output_target_inputs(0).size() == 0) {
                    to_remove_descriptors_indexes[body_idx].push_back(i);
                    pass_required = true;
                } else {
                    // collecting required inputs is needed to detect cases where the input
                    // is not needed in a one body, but the other one uses it (for example If case)
                    required_inputs.insert(op_inputs[body_in_descriptors[i]->m_input_index]);  // only unique
                }
            }
        }
        if (pass_required) {
            using DescType = op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
            auto update_body_param_desc = [](DescType& descriptors, uint64_t removed_body_idx) {
                for (auto& desc : descriptors) {
                    if (desc->m_body_parameter_index > removed_body_idx) {
                        desc->m_body_parameter_index--;
                    }
                }
            };
            auto update_op_inputs_desc = [&subgraphs_size](const std::shared_ptr<op::util::MultiSubGraphOp>& op,
                                                           uint64_t removed_loop_idx) {
                for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
                    auto& descriptors = op->get_input_descriptions(static_cast<int>(body_idx));
                    for (auto& desc : descriptors) {
                        if (desc->m_input_index > removed_loop_idx) {
                            desc->m_input_index--;
                        }
                    }
                }
            };
            // Remove dangling body params and input and update input descriptors
            for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
                auto& body_in_descriptors = multi_subgraph_op->get_input_descriptions(static_cast<int>(body_idx));
                auto& body_func = multi_subgraph_op->get_function(static_cast<int>(body_idx));
                auto& body_params = body_func->get_parameters();
                op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector updated_body_in_descriptors;
                for (size_t desc_idx = 0; desc_idx < body_in_descriptors.size(); ++desc_idx) {
                    if (std::count(std::begin(to_remove_descriptors_indexes[body_idx]),
                                   std::end(to_remove_descriptors_indexes[body_idx]),
                                   desc_idx) > 0) {
                        auto& body_param = body_params[body_in_descriptors[desc_idx]->m_body_parameter_index];
                        body_func->remove_parameter(body_param);
                        // Move all body indexes which are after these indicated by to_remove_descriptors_indexes
                        update_body_param_desc(body_in_descriptors,
                                               body_in_descriptors[desc_idx]->m_body_parameter_index);
                        // remove dangling input of MultiSubGraphOp which was not removed earlier
                        auto& current_input = op_inputs[body_in_descriptors[desc_idx]->m_input_index];
                        if (std::count(std::begin(required_inputs), std::end(required_inputs), current_input) == 0 &&
                            std::count(std::begin(op_inputs), std::end(op_inputs), current_input) > 0) {
                            op_inputs.erase(std::next(op_inputs.begin(), body_in_descriptors[desc_idx]->m_input_index));
                            // Move all input indexes (in all bodies) which are after these indicated by
                            // to_remove_descriptors_indexes and are not used in any body
                            update_op_inputs_desc(multi_subgraph_op, body_in_descriptors[desc_idx]->m_input_index);
                        }
                    } else {
                        updated_body_in_descriptors.emplace_back(body_in_descriptors[desc_idx]);
                    }
                }
                multi_subgraph_op->set_input_descriptions(static_cast<int>(body_idx), updated_body_in_descriptors);
            }
            multi_subgraph_op->set_arguments(op_inputs);
        }
        return false;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(multi_subgraph_op_pattern, matcher_name);
    this->register_matcher(m, callback);
}
