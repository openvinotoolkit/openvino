// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/stateful_to_stateless.hpp"

#include <regex>
#include <string>

#include "openvino/cc/pass/itt.hpp"
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
        const std::regex naming_convention =
            std::regex(R"((past_key_values\.(\d+)\.(key|value))(present\.(\d+)\.(key|value)))");
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
            }
        }
    } else {
        OPENVINO_THROW(
            "Stateful models without `beam_idx` input are not supported in StatefulToStateless transformation");
    }
    model->remove_parameter(beam_idx);

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
        new_parameters.push_back(parameter);
        new_results.push_back(result);
    }

    model->add_parameters(new_parameters);
    model->add_results(new_results);

    return true;
}
