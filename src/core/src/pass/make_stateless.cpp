// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/make_stateless.hpp"

#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/variable.hpp"

bool ov::pass::MakeStateless::selected(const std::string& variable_id) const {
    return m_variable_ids.empty() || m_variable_ids.count(variable_id) > 0;
}

bool ov::pass::MakeStateless::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(MakeStateless);

    // Group readers and writers by variable id. A variable can be read by several ReadValue
    // operations (all of them produce the same state value) and written by at most one Assign
    // operation; anything else cannot be represented without explicit state merge logic.
    std::map<std::string, std::vector<std::shared_ptr<ov::op::util::ReadValueBase>>> reads;
    for (const auto& op : model->get_ops()) {
        auto read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(op);
        if (!read_value || !selected(read_value->get_variable_id())) {
            continue;
        }
        reads[read_value->get_variable_id()].push_back(read_value);
    }

    std::map<std::string, std::shared_ptr<ov::op::util::AssignBase>> writes;
    for (const auto& sink : model->get_sinks()) {
        auto assign = ov::as_type_ptr<ov::op::util::AssignBase>(sink);
        if (!assign || !selected(assign->get_variable_id())) {
            continue;
        }
        OPENVINO_ASSERT(writes.count(assign->get_variable_id()) == 0,
                        "MakeStateless transformation doesn't support several Assign operations "
                        "for the variable '",
                        assign->get_variable_id(),
                        "'.");
        writes[assign->get_variable_id()] = assign;
    }

    ov::ParameterVector new_parameters;
    ov::ResultVector new_results;

    for (const auto& read : reads) {
        const auto& variable_id = read.first;
        const auto& read_values = read.second;

        // All ReadValue operations of one variable represent the same state value: all of them
        // are replaced by a single Parameter named after the variable id.
        const auto& first = read_values.front();
        for (const auto& read_value : read_values) {
            OPENVINO_ASSERT(read_value->get_output_element_type(0) == first->get_output_element_type(0) &&
                                read_value->get_output_partial_shape(0) == first->get_output_partial_shape(0),
                            "MakeStateless transformation doesn't support ReadValue operations of the variable '",
                            variable_id,
                            "' with different types or shapes.");
        }

        auto parameter = std::make_shared<ov::op::v0::Parameter>(first->get_output_element_type(0),
                                                                 first->get_output_partial_shape(0));
        ov::op::util::set_name(*parameter, variable_id);
        ov::copy_runtime_info(first, parameter);
        for (const auto& read_value : read_values) {
            ov::replace_node(read_value, parameter);
        }
        new_parameters.push_back(parameter);

        // The state value written by the corresponding Assign is exposed as a Result, keeping
        // the produced data reachable after the sink is removed.
        const auto& write = writes.find(variable_id);
        if (write == writes.end()) {
            continue;
        }
        auto result = std::make_shared<ov::op::v0::Result>(write->second->input_value(0));
        ov::copy_runtime_info(write->second, result);
        model->remove_sink(write->second);
        new_results.push_back(result);
    }

    // Variables that are written but never read cannot keep the written data reachable either:
    // their Assign input value is exposed as a Result as well.
    for (const auto& write : writes) {
        if (reads.count(write.first) > 0) {
            continue;
        }
        auto result = std::make_shared<ov::op::v0::Result>(write.second->input_value(0));
        ov::copy_runtime_info(write.second, result);
        model->remove_sink(write.second);
        new_results.push_back(result);
    }

    for (const auto& read : reads) {
        auto variable = model->get_variable_by_id(read.first);
        if (variable) {
            model->remove_variable(variable);
        }
    }
    for (const auto& write : writes) {
        auto variable = model->get_variable_by_id(write.first);
        if (variable) {
            model->remove_variable(variable);
        }
    }

    model->add_parameters(new_parameters);
    model->add_results(new_results);
    return true;
}
