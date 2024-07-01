// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/stateful_to_stateless.hpp"

#include <regex>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

namespace {

std::shared_ptr<ov::Node> setName(std::shared_ptr<ov::Node> node, const std::string& name) {
    // Set name for both node and output tensor (should be only one tensor, and any other names will be overriden by a
    // given single name)
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() == 1);
    node->get_output_tensor(0).set_names({name});
    return node;
}


// Templated method that has the same effect as not templated `setName` but saves Op type for convenient calls chaining
template <typename T>
inline std::shared_ptr<T> setName(std::shared_ptr<T> node, const std::string& name) {
    setName(std::dynamic_pointer_cast<ov::Node>(node), name);
    return node;
}


std::shared_ptr<v0::Parameter> get_parameter_by_tensor_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_output_tensor(0).get_names().count(name))
            return param;
    }
    return nullptr;     // nullptr and return type are only difference from ov::Model::input(name)
}


struct Variable {
    struct Context {
        // to hold a compiled once regex for all Variable instances
        const std::regex naming_convention = std::regex(R"((past_key_values\.(\d+)\.(key)|(value))(present\.(\d+)\.(key)|(value)))");
    };

    Variable(const Context& context, const std::string& variable_name) : context(context), variable_name(variable_name) {
        // Try to decode original naming of the corresponding input and output in the stateless model
        std::smatch match;
        if(std::regex_match(variable_name, match, context.naming_convention))
        {
            std::cerr << "MATCHED\n";
        }

        input_name = "input_restored." + variable_name;
        output_name = "output_restored." + variable_name;
    }

    const Context& context;
    size_t index;   // layer index
    std::string variable_name;
    std::string input_name;     // restored name of input in the stateless model, empty if name is not recognized
    std::string output_name;
};

typedef std::vector<Variable> Variables;


void restore_kv_cache_order(Variables& variables, const std::shared_ptr<ov::Model>& model) {
    // Try to restore variable order based on the known naming convention from optimum-intel
    // If names are not satisfy the expected convention, fallback to use order of Assigns in the model->get_sinks()

    // TODO...
}

} // namespace


bool ov::pass::StatefulToStateless::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(StatefulToStateless);

    auto beam_idx = get_parameter_by_tensor_name(model, "beam_idx");
    Variables variables; // to collect variables corresponding to future_params
    variables.reserve(model->get_sinks().size());
    Variable::Context context;
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> future_params;  // to collect nodes, each with a single output that will be replaced by new parameters
    if(beam_idx) {
        for(const ov::Input<ov::Node>& input: beam_idx->get_output_target_inputs(0)) {
            if(auto gather = std::dynamic_pointer_cast<op::util::GatherBase>(input.get_node()->shared_from_this())) {
                auto read_value = std::dynamic_pointer_cast<op::util::ReadValueBase>(gather->get_input_node_shared_ptr(0));
                OPENVINO_ASSERT(
                    read_value,
                    "Unexpected model topology in StatefulToStateless: no ReadValue is found at the first input of Gather by `beam_idx` parameter");
                auto variable_name = read_value->get_variable_id();
                variables.push_back(Variable(context, variable_name));
                future_params[variable_name] = gather;
            }
        }
    } else {
        OPENVINO_THROW("Stateful models without `beam_idx` input are not supported in StatefulToStateless transformation");
    }
    model->remove_parameter(beam_idx);

    restore_kv_cache_order(variables, model);

    typedef std::shared_ptr<op::util::AssignBase> PAssign;
    std::unordered_map<std::string, PAssign> assigns_by_var_id;
    for(auto sink: model->get_sinks()) {
        if(auto assign = std::dynamic_pointer_cast<op::util::AssignBase>(sink)) {
            assigns_by_var_id[assign->get_variable_id()] = assign;
        }
    }

    ov::ParameterVector new_parameters;
    ov::ResultVector new_results;
    new_parameters.reserve(variables.size());
    new_results.reserve(variables.size());

    for(const auto& variable_id: variables) {
        auto future_param = future_params[variable_id.variable_name];
        auto parameter = setName(std::make_shared<v0::Parameter>(
                future_param->get_output_element_type(0),
                future_param->get_output_partial_shape(0)),
            variable_id.input_name);

        replace_node(future_param, parameter);

        auto assign = assigns_by_var_id[variable_id.variable_name];
        auto result = setName(std::make_shared<v0::Result>(
                assign->input_value(0)),
            variable_id.output_name);

        model->remove_sink(assign);  // Don't do replace_node(assign, result)! It will lead to silently incorrect model.
        model->remove_variable(model->get_variable_by_id(variable_id.variable_name));
        new_parameters.push_back(parameter);
        new_results.push_back(result);
    }

    model->add_parameters(new_parameters);
    model->add_results(new_results);

    return true;
}
