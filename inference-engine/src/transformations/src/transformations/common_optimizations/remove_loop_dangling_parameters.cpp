// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_loop_dangling_parameters.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <openvino/opsets/opset8.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <ngraph/rt_info.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ov::pass::RemoveLoopDanglingParameters, "RemoveLoopDanglingParameters", 0);

ov::pass::RemoveLoopDanglingParameters::RemoveLoopDanglingParameters() {
    MATCHER_SCOPE(RemoveLoopDanglingParameters);
    auto loop_pattern = pattern::wrap_type<opset8::Loop>();
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto loop = std::dynamic_pointer_cast<opset8::Loop>(m.get_match_root());
        auto& body_func = loop->get_function();
        auto loop_inputs = loop->input_values();
        bool pass_applied = false;

        auto& body_inputs_descriptors = loop->get_input_descriptions();
        using DescType = decltype(body_inputs_descriptors);
        auto update_descriptors = [](DescType& descriptors, uint64_t removed_body_idx, uint64_t removed_loop_idx){
        for (auto& desc : descriptors) {
            if (desc->m_body_parameter_index > removed_body_idx) {
                desc->m_body_parameter_index--;
            }
            if (desc->m_input_index > removed_loop_idx) {
                desc->m_input_index--;
            }
        }};

        for (auto desc_it = body_inputs_descriptors.begin(); desc_it != body_inputs_descriptors.end();) {
            auto& body_params = body_func->get_parameters();
            auto body_param = body_params[(*desc_it)->m_body_parameter_index];
            if (body_param->get_output_target_inputs(0).size() == 0) {
                body_func->remove_parameter(body_param);
                loop_inputs.erase(loop_inputs.begin()+(*desc_it)->m_input_index);
                // Move all next descriptors after removing input
                update_descriptors(body_inputs_descriptors, (*desc_it)->m_body_parameter_index, (*desc_it)->m_input_index);
                desc_it = body_inputs_descriptors.erase(desc_it);
                pass_applied = true;
            } else {
                ++desc_it;
            }
        }
        if (pass_applied) {
            loop->set_input_descriptions(0, body_inputs_descriptors);
            loop->set_arguments(loop_inputs);
        }
        return pass_applied;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(loop_pattern, matcher_name);
    this->register_matcher(m, callback);
}
