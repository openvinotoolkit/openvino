// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/low_latency.hpp"

#include <memory>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::LowLatency, "LowLatency", 0);

ngraph::pass::LowLatency::LowLatency() {
    auto tensor_iterator = ngraph::pattern::wrap_type<opset5::TensorIterator>();
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto ti = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator> (m.get_match_root());
        if (!ti) {
            return false;
        }

        std::vector<std::shared_ptr<ngraph::op::Sink>> assigns;
        const auto& func = ti->get_function();
        for (const auto& in : ti->get_input_descriptions()) {
            int64_t variable_id = 0;
            // Process all back edges
            if (const auto& merged_in = std::dynamic_pointer_cast<ngraph::opset5::TensorIterator::MergedInputDescription>(in)) {
                // Insert ReadValue nodes: Parameter -> (new ReadValue) -> consumers
                const std::string variable_name(ti->get_friendly_name() + "/variable" + std::to_string(variable_id));
                const auto& inputs_to = func->get_parameters().at(merged_in->m_body_parameter_index)->get_output_target_inputs(0);
                auto read_value = std::make_shared<opset5::ReadValue>(func->get_parameters().at(merged_in->m_body_parameter_index),
                                                                      variable_name);
                for (const auto& input_to : inputs_to) {
                    input_to.replace_source_output(read_value->output(0));
                }

                // insert Assign nodes: provider -> (new Assign) -> Result
                const auto res = func->get_results().at(merged_in->m_body_value_index);
                auto assign = std::make_shared<opset5::Assign>(res->input_value(0), variable_name);

                // control dependency so that ReadValue is processed before Assign
                assign->add_control_dependency(read_value);
                assigns.emplace_back(assign);
            }
        }
        // save Assign in the func so that it gets into graph traversals and isn't deleted.
        func->add_sinks(assigns);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tensor_iterator, "LowLatency");
    register_matcher(m, callback);
}
