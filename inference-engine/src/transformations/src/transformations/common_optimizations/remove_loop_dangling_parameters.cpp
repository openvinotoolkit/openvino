// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_loop_dangling_parameters.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::RemoveLoopDanglingParameters, "RemoveLoopDanglingParameters", 0);

ngraph::pass::RemoveLoopDanglingParameters::RemoveLoopDanglingParameters() {
    MATCHER_SCOPE(RemoveLoopDanglingParameters);
    auto loop_pattern = pattern::wrap_type<opset8::Loop>();
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto loop = std::dynamic_pointer_cast<opset8::Loop>(m.get_match_root());
        auto& body_inputs_descriptors = loop->get_input_descriptions();
        auto body_func = loop->get_function();
        auto body_params = body_func->get_parameters();
        auto loop_inputs = loop->input_values();
        bool pass_applied = false;
        for (auto desc_it = body_inputs_descriptors.begin(); desc_it != body_inputs_descriptors.end(); ++desc_it) {
            auto body_param = body_params[(*desc_it)->m_body_parameter_index];
            if (body_param->get_output_target_inputs(0).size() == 0) {
                body_func->remove_parameter(body_param);
                loop_inputs.erase(loop_inputs.begin()+(*desc_it)->m_input_index);
                body_inputs_descriptors.erase(desc_it);
                // Move all descriptors after removing input
                for (auto it = desc_it; it != body_inputs_descriptors.end(); ++it) {
                    (*it)->m_body_parameter_index--;
                    (*it)->m_input_index--;
                }
                pass_applied = true;
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
