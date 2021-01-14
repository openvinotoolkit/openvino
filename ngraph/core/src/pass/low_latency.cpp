// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pass/low_latency.hpp"

#include <memory>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/variant.hpp>
#include <numeric>

NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency, "LowLatency", 0);

ngraph::pass::LowLatency::LowLatency()
{
    auto tensor_iterator = ngraph::pattern::wrap_type<opset5::TensorIterator>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator>(m.get_match_root());
        if (!ti)
        {
            return false;
        }

        // Mark the TI layer to be unrolled. Enable unconditional ti unrolling for all plugins.
        auto& rt_info = ti->get_rt_info();
        rt_info["UNROLL_TI"] = std::make_shared<ngraph::VariantWrapper<int64_t>>(1);

        int64_t variable_id = 0;
        std::vector<std::shared_ptr<ngraph::op::Sink>> assigns;
        const auto& func = ti->get_function();
        auto in_descs = ti->get_input_descriptions();
        std::vector<int> inputs_ind_to_delete;
        for (const auto& in : in_descs)
        {
            // Process all back edges
            if (const auto& merged_in = std::dynamic_pointer_cast<
                    ngraph::opset5::TensorIterator::MergedInputDescription>(in))
            {
                // Insert ReadValue nodes: Parameter -> (new ReadValue) -> consumers
                const auto& inputs_to = func->get_parameters()
                                            .at(merged_in->m_body_parameter_index)
                                            ->get_output_target_inputs(0);
                const std::string variable_name(ti->get_friendly_name() + "/" +
                                                func->get_parameters()
                                                    .at(merged_in->m_body_parameter_index)
                                                    ->get_friendly_name() +
                                                "/variable_" + std::to_string(variable_id));
                auto init_shape = func->get_parameters().at(merged_in->m_body_parameter_index)->get_partial_shape().get_shape();
                int zeros_length = 1.0;
                for (auto i : init_shape){
                    zeros_length *= i;
                }
                std::vector<float> zeros(zeros_length, 0);
                auto init_const = op::Constant::create(element::f32, init_shape, zeros);
                auto read_value = std::make_shared<opset5::ReadValue>(init_const, variable_name);
                read_value->set_friendly_name(variable_name);
                for (const auto& input_to : inputs_to)
                {
                    input_to.replace_source_output(read_value->output(0));
                }

                // insert Assign nodes: provider -> (new Assign) -> Result
                const auto res = func->get_results().at(merged_in->m_body_value_index);
                auto assign = std::make_shared<opset5::Assign>(res->input_value(0), variable_name);
                // control dependency so that ReadValue is processed before Assign
                assign->add_control_dependency(read_value);
                assigns.emplace_back(assign);
                // save index of input to delete
                inputs_ind_to_delete.push_back(merged_in->m_body_parameter_index);
            }
            variable_id++;
        } 

        std::sort(inputs_ind_to_delete.begin(), inputs_ind_to_delete.end());
        auto params = func->get_parameters();
        for (int i=inputs_ind_to_delete.size()-1; i>=0; i--){
            func->remove_parameter(params[inputs_ind_to_delete[i]]);
        }
        
        // remove replaced inputs from descriptions
        ti->get_input_descriptions().erase(remove_if(ti->get_input_descriptions().begin(), ti->get_input_descriptions().end(), 
        [](ngraph::op::util::InputDescriptionPtr it){
            return std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::MergedInputDescription>(it);
            }), ti->get_input_descriptions().end());
        
        // fix indexes for other inputs
         for (const auto& in : in_descs)
        {
            int already_deleted = 0;
            while(in->m_body_parameter_index > inputs_ind_to_delete[already_deleted]){
                already_deleted++;
            }
            in->m_body_parameter_index -= already_deleted;
        }

        // save Assign in the func so that it gets into graph traversals and isn't deleted.
        func->add_sinks(assigns);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "LowLatency");
    register_matcher(m, callback);
}
