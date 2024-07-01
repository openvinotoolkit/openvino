// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_stateful.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset8.hpp"

namespace {

std::tuple<ov::pass::MakeStateful::ParamResPairs, std::vector<std::string>> find_param_results_by_names(
    const std::shared_ptr<ov::Model>& model,
    const std::map<std::string, std::string>& param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs_to_replace;
    std::vector<std::string> variable_names;
    const auto& params = model->get_parameters();
    const auto& results = model->get_results();

    std::set<ov::Node*> uniq_params;
    std::set<ov::Node*> uniq_res;

    // find corresponding param and result by name and add to the list
    for (const auto& param_res : param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;
        auto param = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<ov::Node>& node) {
            const auto& possible_names = node->output(0).get_names();
            return possible_names.find(param_name) != possible_names.end();
        });
        OPENVINO_ASSERT(param != params.end(),
                        "The tensor name ",
                        param_name,
                        " is not associated with any of "
                        "Parameters in the network.");
        uniq_params.insert(param->get());

        auto res = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<ov::Node>& node) {
            const auto& possible_names = node->output(0).get_names();
            return possible_names.find(res_name) != possible_names.end();
        });

        OPENVINO_ASSERT(res != results.end(),
                        "The tensor name ",
                        res_name,
                        " is not associated with any of "
                        "Results in the network.");

        // In case of several Results connected to one output tensor,
        // We can't determine what result we need to take exactly.
        // But we can take first unused, the order is not important, data is the same.
        ov::op::v0::Result* unused_res = nullptr;
        for (const auto& target_in : (*res)->input_value(0).get_target_inputs()) {
            auto is_target_res = ov::as_type<ov::op::v0::Result>(target_in.get_node());
            if (!is_target_res) {
                continue;
            }
            if (uniq_res.find(is_target_res) == uniq_res.end()) {
                unused_res = is_target_res;
                break;
            }
        }
        OPENVINO_ASSERT(unused_res != nullptr,
                        "All Result operations associated with the tensor ",
                        res_name,
                        " are already involved in the transformation.");
        uniq_res.insert(unused_res);

        if (auto casted = std::dynamic_pointer_cast<ov::op::v0::Result>(unused_res->shared_from_this()))
            pairs_to_replace.emplace_back(*param, casted);
        variable_names.push_back(param_name + res_name);
    }
    return std::make_tuple(pairs_to_replace, variable_names);
}
}  // namespace

bool ov::pass::MakeStateful::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(MakeStateful);
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

    ov::op::util::VariableVector variables;
    SinkVector sinks;

    for (size_t i = 0; i < m_param_res_pairs.size(); ++i) {
        const auto& param = m_param_res_pairs[i].first;
        const auto& res = m_param_res_pairs[i].second;

        // Create Variable
        const auto& var_name = variable_names[i];
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{param->get_partial_shape(), param->get_element_type(), var_name});
        variables.push_back(variable);

        // Create ReadValue
        auto read_val = std::make_shared<ov::op::v6::ReadValue>(variable);
        replace_node(param, read_val);
        ov::copy_runtime_info(param, read_val);

        // Create Assign
        auto assign = std::make_shared<ov::op::v6::Assign>(res->input_value(0), variable);
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
