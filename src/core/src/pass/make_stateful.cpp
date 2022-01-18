// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_stateful.hpp"

#include <memory>
#include <ngraph/rt_info.hpp>
#include <openvino/op/util/variable.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ngraph;
using namespace opset8;
using namespace op::util;

namespace {
string generate_variable_name(const shared_ptr<Parameter>& param, const shared_ptr<Result>& res) {
    return param->get_friendly_name() + res->get_friendly_name();
}

ov::pass::MakeStateful::ParamResPairs find_param_results_by_names(
    const shared_ptr<ngraph::Function>& func,
    const std::map<std::string, std::string>& param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs_to_replace;
    const auto& params = func->get_parameters();
    const auto& results = func->get_results();

    // find corresponding param and result by name and add to the list
    for (const auto& param_res : param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name() == param_name;
        });
        NGRAPH_CHECK(param != params.end(), "Parameter node with name = ", param_name, "doesn't exist in the function");

        auto res = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            return node->get_friendly_name() == res_name;
        });
        NGRAPH_CHECK(res != results.end(), "Result node with name = ", res_name, " doesn't exist in the function");

        pairs_to_replace.emplace_back(*param, *res);
    }
    return pairs_to_replace;
}
}  // namespace

bool ov::pass::MakeStateful::run_on_model(const std::shared_ptr<ov::Model>& f) {
    if (m_param_res_pairs.empty()) {
        m_param_res_pairs = find_param_results_by_names(f, m_param_res_names);
    }

    VariableVector variables;
    SinkVector sinks;
    for (const auto& pair : m_param_res_pairs) {
        const auto& param = pair.first;
        const auto& res = pair.second;

        NGRAPH_CHECK(param->get_partial_shape().is_static(),
                     "Shape of Parameter ",
                     param->get_friendly_name(),
                     " must be static. MakeStateful transformation doesn't support dynamic shapes.");

        // Create Variable
        std::string var_name = generate_variable_name(param, res);
        auto variable =
            std::make_shared<Variable>(VariableInfo{param->get_shape(), param->get_element_type(), var_name});
        variables.push_back(variable);

        // Create ReadValue
        auto const_zero = make_shared<Constant>(param->get_element_type(), param->get_shape(), 0);
        auto read_val = make_shared<ReadValue>(const_zero, variable);
        replace_node(param, read_val);
        copy_runtime_info(param, {read_val, const_zero});

        // Create Assign
        auto assign = make_shared<Assign>(res->input_value(0), variable);
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
