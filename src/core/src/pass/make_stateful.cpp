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

std::tuple<ov::pass::MakeStateful::ParamResPairs, std::vector<std::string>> find_param_results_by_names(
    const shared_ptr<ngraph::Function>& func,
    const std::map<std::string, std::string>& param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs_to_replace;
    std::vector<std::string> variable_names;
    const auto& params = func->get_parameters();
    const auto& results = func->get_results();

    std::set<Node*> uniq_params;
    std::set<Node*> uniq_res;

    // find corresponding param and result by name and add to the list
    for (const auto& param_res : param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;
        auto param = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            const auto& possible_names = node->output(0).get_names();
            return possible_names.find(param_name) != possible_names.end();
        });
        NGRAPH_CHECK(param != params.end(),
                     "The tensor name ",
                     param_name,
                     " is not associated with any of "
                     "Parameters in the network.");
        uniq_params.insert(param->get());

        auto res = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<ngraph::Node>& node) {
            const auto& possible_names = node->output(0).get_names();
            return possible_names.find(res_name) != possible_names.end();
        });

        NGRAPH_CHECK(res != results.end(),
                     "The tensor name ",
                     res_name,
                     " is not associated with any of "
                     "Results in the network.");

        // In case of several Results connected to one output tensor,
        // We can't determine what result we need to take exactly.
        // But we can take first unused, the order is not important, data is the same.
        opset8::Result* unused_res = nullptr;
        for (const auto& target_in : (*res)->input_value(0).get_target_inputs()) {
            auto is_target_res = ov::as_type<opset8::Result>(target_in.get_node());
            if (!is_target_res) {
                continue;
            }
            if (uniq_res.find(is_target_res) == uniq_res.end()) {
                unused_res = is_target_res;
                break;
            }
        }
        NGRAPH_CHECK(unused_res != nullptr,
                     "All Result operations associated with the tensor ",
                     res_name,
                     " are already involved in the transformation.");
        uniq_res.insert(unused_res);

        if (auto casted = std::dynamic_pointer_cast<opset8::Result>(unused_res->shared_from_this()))
            pairs_to_replace.emplace_back(*param, casted);
        variable_names.push_back(param_name + res_name);
    }
    return std::make_tuple(pairs_to_replace, variable_names);
}
}  // namespace

bool ov::pass::MakeStateful::run_on_model(const std::shared_ptr<ov::Model>& f) {
    // in case of user passes the tensor names to find Parameter/Result nodes, we use these tensor names
    // to generate variable names. In case of user passes Parameter/Result nodes directly, we use friendly
    // names of these nodes to generate variable names.
    std::vector<std::string> variable_names;
    if (m_param_res_pairs.empty()) {
        std::tie(m_param_res_pairs, variable_names) = find_param_results_by_names(f, m_param_res_names);
    } else {
        for (const auto& pair : m_param_res_pairs) {
            variable_names.push_back(pair.first->get_friendly_name() + pair.second->get_friendly_name());
        }
    }

    VariableVector variables;
    SinkVector sinks;

    for (size_t i = 0; i < m_param_res_pairs.size(); ++i) {
        const auto& param = m_param_res_pairs[i].first;
        const auto& res = m_param_res_pairs[i].second;

        NGRAPH_CHECK(param->get_partial_shape().is_static(),
                     "Shape of Parameter ",
                     param->get_friendly_name(),
                     " must be static. MakeStateful transformation doesn't support dynamic shapes.");

        // Create Variable
        std::string var_name = variable_names[i];
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
