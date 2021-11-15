// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <openvino/opsets/opset8.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <ngraph/rt_info.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ov::pass::RemoveMultiSubGraphOpDanglingParams, "RemoveMultiSubGraphOpDanglingParams", 0);

ov::pass::RemoveMultiSubGraphOpDanglingParams::RemoveMultiSubGraphOpDanglingParams() {
    MATCHER_SCOPE(RemoveMultiSubGraphOpDanglingParams);
    auto multi_subgraph_op_pattern = pattern::wrap_type<
        op::util::MultiSubGraphOp, opset8::Loop, opset8::TensorIterator, opset8::If>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(m.get_match_root());
        if (multi_subgraph_op == nullptr) {
            return false;
        }
        bool pass_required = false;
        for (size_t body_idx=0; body_idx < multi_subgraph_op->get_internal_subgraphs_size(); ++body_idx) {
            auto& body_func = multi_subgraph_op->get_function(body_idx);
            auto& body_params = body_func->get_parameters();
            auto& body_in_descriptors = multi_subgraph_op->get_input_descriptions(body_idx);
            std::vector<size_t> to_remove_descriptors_indexes;

            for (size_t i = 0; i < body_in_descriptors.size(); ++i) {
                auto& body_param = body_params[body_in_descriptors[i]->m_body_parameter_index];
                if (body_param->get_output_target_inputs(0).size() == 0) {
                    to_remove_descriptors_indexes.push_back(i);
                    pass_required = true;
                }
            }
            if (pass_required) {
                auto updated_inputs = multi_subgraph_op->input_values();
                op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector updated_body_in_descriptors;
                for (size_t i = 0; i < body_in_descriptors.size(); ++i) {
                    if (std::count(std::begin(to_remove_descriptors_indexes), std::end(to_remove_descriptors_indexes), i) > 0) {
                        auto& body_param = body_params[body_in_descriptors[i]->m_body_parameter_index];
                        body_func->remove_parameter(body_param);
                        updated_inputs.erase(std::next(updated_inputs.begin(), body_in_descriptors[i]->m_input_index));
                        // Move all descriptors which are after these indicated by to_remove_descriptors_indexes
                        for (size_t j=i+1; j < body_in_descriptors.size(); ++j) {
                            body_in_descriptors[j]->m_body_parameter_index--;
                            body_in_descriptors[j]->m_input_index--;
                        }
                    } else {
                        updated_body_in_descriptors.emplace_back(body_in_descriptors[i]);
                    }
                }
                multi_subgraph_op->set_input_descriptions(body_idx, updated_body_in_descriptors);
                multi_subgraph_op->set_arguments(updated_inputs);
            }
        }
        return pass_required;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(multi_subgraph_op_pattern, matcher_name);
    this->register_matcher(m, callback);
}
