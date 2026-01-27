// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/stateful_to_stateless.hpp"

#include <regex>
#include <string>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

namespace {
std::shared_ptr<v0::Parameter> get_parameter_by_tensor_name(const std::shared_ptr<ov::Model>& model,
                                                            const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_output_tensor(0).get_names().count(name))
            return param;
    }
    return nullptr;  // nullptr and return type are only difference from ov::Model::input(name)
}

struct Variable {
    struct Context {
        // to hold compiled once regex for all Variable instances
        // Pattern 1: optimum-intel convention (e.g., past_key_values.0.keypresent.0.key)
        const std::regex naming_convention =
            std::regex(R"((past_key_values\.(\d+)\.(key|value))(present\.(\d+)\.(key|value)))");
        // Pattern 2: LFM2 convention (e.g., cache_params.past.key.0cache_params.present.key.0)
        const std::regex lfm2_naming_convention =
            std::regex(R"(cache_params\.past\.(key|value|conv)\.(\d+)cache_params\.present\.(key|value|conv)\.(\d+))");
    };

    Variable(const Context& context, const std::string& variable_name) : variable_name(variable_name) {
        // Try to decode original naming of the corresponding input and output in the stateless model
        std::smatch match;
        if (std::regex_match(variable_name, match, context.naming_convention)) {
            input_name = match[1].str();
            output_name = match[4].str();
            auto input_index = match[2].str();
            auto output_index = match[5].str();
            if (input_index == output_index && input_index.length() <= std::numeric_limits<int>::digits10) {
                index = std::stoi(input_index) * 2 + int(match[3].str() == "value");  // order key before value
            } else {
                index = -1;
            }
        } else if (std::regex_match(variable_name, match, context.lfm2_naming_convention)) {
            // LFM2 pattern: cache_params.past.key.0cache_params.present.key.0
            std::string past_type = match[1].str();
            std::string past_idx = match[2].str();
            std::string present_type = match[3].str();
            std::string present_idx = match[4].str();

            input_name = "past_key_values." + past_idx + "." + past_type;
            output_name = "present." + present_idx + "." + present_type;

            if (past_idx == present_idx && past_type == present_type &&
                past_idx.length() <= std::numeric_limits<int>::digits10) {
                // For "conv" type, place them after key/value pairs
                int type_offset = (past_type == "key") ? 0 : (past_type == "value") ? 1 : 2;
                index = std::stoi(past_idx) * 3 + type_offset;  // key=0, value=1, conv=2 within each layer
            } else {
                index = -1;
            }
        } else {
            // Variable name doesn't follow the expected naming convention. It doens't prevent forming
            // a correct stateless model but doesn't give a way to restore all names and inputs/outputs ordering
            // accurately.
            input_name = "input_restored." + variable_name;
            output_name = "output_restored." + variable_name;
            index = -1;
        }
    }

    int index;                  // layer index, -1 means the index isn't known
    std::string variable_name;  // original variable_id
    std::string input_name;     // restored name of input
    std::string output_name;    // restored name of output
};

typedef std::vector<Variable> Variables;

void restore_kv_cache_order(Variables& variables, const std::unordered_map<std::string, size_t>& var_index_by_var_id) {
    // Try to restore variable order based on the known naming convention from optimum-intel.
    // If names are not satisfy the expected convention, fallback to use order based on var_index_by_var_id
    // Sort items that do satisfy the naming conventions before items that don't satisfy.

    std::stable_sort(variables.begin(), variables.end(), [&](const Variable& a, const Variable& b) {
        if (a.index >= 0 && b.index >= 0) {
            return a.index < b.index;
        } else if (a.index >= 0 && b.index < 0) {
            return true;
        } else if (a.index < 0 && b.index >= 0) {
            return false;
        } else {  // a.index < 0 && b.index < 0
            return var_index_by_var_id.at(a.variable_name) < var_index_by_var_id.at(b.variable_name);
        }
    });
}

}  // namespace

bool ov::pass::StatefulToStateless::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(StatefulToStateless);

    auto beam_idx = get_parameter_by_tensor_name(model, "beam_idx");
    Variables variables;  // to collect variables corresponding to future_params
    variables.reserve(model->get_sinks().size());
    Variable::Context context;
    std::unordered_map<std::string, std::shared_ptr<ov::Node>>
        future_params;  // to collect nodes, each with a single output that will be replaced by new parameters
    std::unordered_set<std::string> processed_variable_ids;  // Track which ReadValues we've processed
    if (beam_idx) {
        for (const ov::Input<ov::Node>& input : beam_idx->get_output_target_inputs(0)) {
            if (auto gather = ov::as_type_ptr<op::util::GatherBase>(input.get_node()->shared_from_this())) {
                auto read_value = ov::as_type_ptr<op::util::ReadValueBase>(gather->get_input_node_shared_ptr(0));
                OPENVINO_ASSERT(read_value,
                                "Unexpected model topology in StatefulToStateless: no ReadValue is found at the first "
                                "input of Gather by `beam_idx` parameter");
                auto variable_name = read_value->get_variable_id();
                variables.push_back(Variable(context, variable_name));
                future_params[variable_name] = gather;
                processed_variable_ids.insert(variable_name);
            }
        }
        model->remove_parameter(beam_idx);
    } else {
        OPENVINO_THROW(
            "Stateful models without `beam_idx` input are not supported in StatefulToStateless transformation");
    }

    // Process ReadValues that are NOT connected via beam_idx: Conv caches in LFM2
    for (const auto& op : model->get_ops()) {
        if (auto read_value = ov::as_type_ptr<op::util::ReadValueBase>(op)) {
            auto variable_name = read_value->get_variable_id();
            std::smatch match;
            if (std::regex_match(variable_name, match, context.lfm2_naming_convention) &&
                (processed_variable_ids.find(variable_name) == processed_variable_ids.end())) {
                variables.push_back(Variable(context, variable_name));
                future_params[variable_name] = read_value;
                processed_variable_ids.insert(variable_name);
            }
        }
    }

    typedef std::shared_ptr<op::util::AssignBase> PAssign;
    std::unordered_map<std::string, PAssign> assigns_by_var_id;
    std::unordered_map<std::string, size_t> assign_index_by_var_id;
    const auto& sinks = model->get_sinks();
    for (size_t i = 0; i < sinks.size(); ++i) {
        if (auto assign = ov::as_type_ptr<op::util::AssignBase>(sinks[i])) {
            const auto& var_id = assign->get_variable_id();
            assigns_by_var_id[var_id] = std::move(assign);
            assign_index_by_var_id[var_id] = i;
        }
    }

    restore_kv_cache_order(variables, assign_index_by_var_id);

    ov::ParameterVector new_parameters;
    ov::ResultVector new_results;
    new_parameters.reserve(variables.size());
    new_results.reserve(variables.size());

    for (const auto& variable_id : variables) {
        auto future_param = future_params[variable_id.variable_name];
        auto parameter = std::make_shared<v0::Parameter>(future_param->get_output_element_type(0),
                                                         future_param->get_output_partial_shape(0));
        ov::op::util::set_name(*parameter, variable_id.input_name);

        replace_node(future_param, parameter);

        auto assign = assigns_by_var_id[variable_id.variable_name];
        auto result = std::make_shared<v0::Result>(assign->input_value(0));
        ov::op::util::set_name(*result, variable_id.output_name);

        model->remove_sink(assign);  // Don't do replace_node(assign, result)! It will lead to silently incorrect model.
        model->remove_variable(model->get_variable_by_id(variable_id.variable_name));
        new_parameters.push_back(std::move(parameter));
        new_results.push_back(std::move(result));
    }

    model->add_parameters(new_parameters);
    model->add_results(new_results);

    return true;
}
