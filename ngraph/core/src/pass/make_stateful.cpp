// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset8.hpp>

#include "openvino/pass/make_stateful_test.hpp"

NGRAPH_RTTI_DEFINITION(ov::pass::MakeStateful, "MakeStateful", 0);

using namespace std;
using namespace ngraph;
using namespace opset8;
using namespace op::util;

namespace {
string generate_variable_name(const shared_ptr<Parameter>& param, const shared_ptr<Result>& res, uint64_t idx) {
    return param->get_friendly_name() + res->get_friendly_name() + std::to_string(idx);
}
}  // namespace

bool ov::pass::MakeStateful::run_on_function(std::shared_ptr<ngraph::Function> f) {
    VariableVector variables;
    SinkVector sinks;
    uint64_t idx = 0;
    for (const auto& pair : m_pairs_to_replace) {
        const auto& param = pair.first;
        const auto& res = pair.second;

        NGRAPH_CHECK(param->get_partial_shape().is_static(),
                     "Shape of Parameter ",
                     param->get_friendly_name(),
                     " must be static. MakeStateful transformation doesn't support dynamic shapes.");

        const auto& target_inputs = param->get_output_target_inputs(0);

        // Create Variable
        std::string var_name = generate_variable_name(param, res, idx++);
        auto variable = std::make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, var_name});
        variables.push_back(variable);

        // Create ReadValue
        auto const_zero = make_shared<Constant>(param->get_element_type(), param->get_shape(), 0);
        auto read_val = make_shared<ReadValue>(const_zero, variable);
        for (const auto& target_in : target_inputs) {
            target_in.replace_source_output(read_val->output(0));
        }
        copy_runtime_info(param, read_val);

        // Create Assign
        const auto& input_to_res = res->input_value(0);
        input_to_res.remove_target_input(res->input(0));
        auto assign = make_shared<Assign>(input_to_res, variable);
        copy_runtime_info(res, assign);

        // Update Function
        sinks.push_back(assign);
        f->remove_result(res);
        f->remove_parameter(param);
        assign->add_control_dependency(read_val);
    }
    f->add_variables(variables);
    f->add_sinks(sinks);
    return true;
}

ov::pass::MakeStateful::InOutPairs ov::pass::MakeStateful::findInputsOutputsByName(
    const shared_ptr<ngraph::Function>& func,
    const vector<std::pair<std::string, std::string>>& param_res_names) {
    InOutPairs pairs_to_replace;
    const auto& params = func->get_parameters();
    const auto& results = func->get_results();

    // find corresponding param and result by name and add to the list
    for (const auto& param_res : param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name().find(param_name) != std::string::npos;
        });
        NGRAPH_CHECK(param != params.end(),
                     "Parameter node with name = ",
                     param_name,
                     "doesn't belong to the function");

        auto res = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name().find(res_name) != std::string::npos;
        });
        NGRAPH_CHECK(res != results.end(), "Result node with name = ", res_name, " doesn't belong to the function");

        pairs_to_replace.emplace_back(*param, *res);
    }
    return pairs_to_replace;
}
