// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/stateful_to_stateless.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

static std::shared_ptr<ov::Node> setName(std::shared_ptr<ov::Node> node, const std::string& name) {
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


static std::shared_ptr<v0::Parameter> get_parameter_by_tensor_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_output_tensor(0).get_names().count(name))
            return param;
    }
    return nullptr;     // nullptr and return type are only difference from ov::Model::input(name)
}


typedef std::string VariableID;
typedef std::vector<VariableID> VariableIDs;


static void restore_kv_cache_order(VariableIDs& variables, const std::shared_ptr<ov::Model>& model) {
    // Try to restore variable order based on the known naming convention from optimum-intel
    // If names are not satisfy the expected convention, fallback to use order of Assigns in the model->get_sinks()

    // TODO...
}


static std::string variable_id_to_input_name(const VariableID variable_id) {
    // TODO: Restore original input name based on optimum-intel convention
    return "input_restored." + variable_id;
}


static std::string variable_id_to_output_name(const VariableID variable_id) {
    // TODO: Restore original output name based on optimum-intel convention
    return "output_restored." + variable_id;
}


bool ov::pass::StatefulToStateless::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(StatefulToStateless);

    auto beam_idx = get_parameter_by_tensor_name(model, "beam_idx");
    typedef std::string VariableID;
    VariableIDs variable_ids; // to collect variables corresponding to future_params
    std::unordered_map<VariableID, std::shared_ptr<ov::Node>> future_params;  // to collect nodes, each with a single output that will be replaced by new parameters
    if(beam_idx) {
        for(const ov::Input<ov::Node>& input: beam_idx->get_output_target_inputs(0)) {
            if(auto gather = std::dynamic_pointer_cast<op::util::GatherBase>(input.get_node()->shared_from_this())) {
                auto read_value = std::dynamic_pointer_cast<op::util::ReadValueBase>(gather->get_input_node_shared_ptr(0));
                OPENVINO_ASSERT(
                    read_value,
                    "Unexpected model topology in StatefulToStateless: no ReadValue is found at the first input of Gather by `beam_idx` parameter");
                auto variable_id = read_value->get_variable_id();
                variable_ids.push_back(variable_id);
                future_params[variable_id] = gather;
            }
        }
    } else {
        OPENVINO_THROW("Stateful models without `beam_idx` input are not supported in StatefulToStateless transformation");
    }
    model->remove_parameter(beam_idx);

    restore_kv_cache_order(variable_ids, model);

    typedef std::shared_ptr<op::util::AssignBase> PAssign;
    std::unordered_map<VariableID, PAssign> assigns_by_var_id;
    for(auto sink: model->get_sinks()) {
        if(auto assign = std::dynamic_pointer_cast<op::util::AssignBase>(sink)) {
            assigns_by_var_id[assign->get_variable_id()] = assign;
        }
    }

    ov::ParameterVector new_parameters;
    ov::ResultVector new_results;
    new_parameters.reserve(variable_ids.size());
    new_results.reserve(variable_ids.size());

    for(const auto& variable_id: variable_ids) {
        auto future_param = future_params[variable_id];
        auto parameter = setName(std::make_shared<v0::Parameter>(
                future_param->get_output_element_type(0),
                future_param->get_output_partial_shape(0)),
            variable_id_to_input_name(variable_id));

        replace_node(future_param, parameter);

        auto assign = assigns_by_var_id[variable_id];
        auto result = setName(std::make_shared<v0::Result>(
                assign->input_value(0)),
            variable_id_to_output_name(variable_id));

        model->remove_sink(assign);  // Don't do replace_node(assign, result)! It will lead to silently incorrect model.
        model->remove_variable(model->get_variable_by_id(variable_id));
        new_parameters.push_back(parameter);
        new_results.push_back(result);
    }

    model->add_parameters(new_parameters);
    model->add_results(new_results);

    return true;
}
