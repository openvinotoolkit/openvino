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

static std::shared_ptr<v0::Parameter> setName(std::shared_ptr<v0::Parameter> node, const std::string& name) {
    // Set name for both node and output tensor (should be only one tensor, and any other names will be overriden by a
    // given single name)
    node->set_friendly_name(name);
    OPENVINO_ASSERT(node->get_output_size() == 1);
    node->get_output_tensor(0).set_names({name});
    return node;
}


std::shared_ptr<v0::Parameter> get_parameter_by_tensor_name(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& param : model->get_parameters()) {
        if (param->get_output_tensor(0).get_names().count(name))
            return param;
    }
    return nullptr;     // nullptr and return type are only difference from ov::Model::input(name)
}


bool ov::pass::StatefulToStateless::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(StatefulToStateless);

    auto beam_idx = get_parameter_by_tensor_name(model, "beam_idx");
    ov::NodeVector future_params;  // to collect nodes, each with a single output that will be replaced by new parameters
    std::vector<std::shared_ptr<op::util::Variable>> variables; // to collect variables corresponding to future_params
    if(beam_idx) {
        for(const ov::Input<ov::Node>& input: beam_idx->get_output_target_inputs(0)) {
            if(auto gather = std::dynamic_pointer_cast<op::util::GatherBase>(input.get_node()->shared_from_this())) {
                auto read_value = std::dynamic_pointer_cast<op::util::ReadValueBase>(gather->get_input_node_shared_ptr(0));
                OPENVINO_ASSERT(
                    read_value,
                    "Unexpected model topology in StatefulToStateless: no ReadValue is found at the first input of Gather by `beam_idx` parameter");
                future_params.push_back(gather);
                variables.push_back(read_value->get_variable());
            }
        }
    } else {
        OPENVINO_THROW("Stateful models without `beam_idx` input are not supported in StatefulToStateless transformation");
    }

    // TODO: Use naming convention for variables to sort them in the orignal order and assing name for new Parameter's/Result's

    std::unordered_map<std::string, std::shared_ptr<op::util::AssignBase>> assign_by_var_name;
    for(auto sink: model->get_sinks()) {
        if(auto assign = std::dynamic_pointer_cast<op::util::AssignBase>(sink)) {
            assign_by_var_name[assign->get_variable_id()] = assign;
        }
    }

    model->remove_parameter(beam_idx);
    for(size_t i = 0; i < future_params.size(); ++i) {
        auto future_param = future_params[i];
        std::string variable_id = variables[i]->get_info().variable_id;
        auto parameter = setName(std::make_shared<v0::Parameter>(
            future_param->get_output_element_type(0),
            future_param->get_output_partial_shape(0)),
            variable_id + ".restored_input");
        model->add_parameters({parameter});
        replace_node(future_param, parameter);
        auto assign = assign_by_var_name[variable_id];
        auto result = std::make_shared<v0::Result>(assign->input_value(0));
        replace_node(assign, result);
        model->remove_sink(assign);
    }

    return true;
}
