// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "input_model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "tf_framework_node.hpp"
#include "utils.hpp"

using namespace ov::frontend::tensorflow;

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::shared_ptr<TranslatorDictionaryType>& translator_map,
                                   const std::string& model_name,
                                   bool failed_fast,
                                   bool telemetry)
    : m_failed_fast(failed_fast),
      m_telemetry(telemetry),
      m_input_model(input_model),
      m_translator_map(translator_map),
      m_model_name(model_name),
      m_ov_model(nullptr),
      m_cached_body_models(std::make_shared<CachedBodyModelsType>()),
      m_telemetry_data(std::make_shared<TelemetryDataType>()) {}

std::shared_ptr<ov::Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    translate_graph(m_input_model, m_ov_model);
    return m_ov_model;
}

std::shared_ptr<TelemetryDataType> TranslateSession::get_telemetry_data() const {
    return m_telemetry_data;
}

void TranslateSession::inject_body_model(std::shared_ptr<ov::Model> body_model,
                                         const std::string& operation_type,
                                         const ov::OutputVector& ov_inputs,
                                         ov::OutputVector& ov_outputs) {
    ov_outputs.clear();
    auto body_parameters = body_model->get_parameters();
    FRONT_END_GENERAL_CHECK(body_parameters.size() == ov_inputs.size(),
                            "[TensorFlow Error] Internal error or incorrect input models: number of "
                            "inputs and arguments to the function " +
                                operation_type + " do not match.");
    for (size_t param_ind = 0; param_ind < body_parameters.size(); ++param_ind) {
        body_parameters[param_ind]->output(0).replace(ov_inputs[param_ind]);
    }
    for (const auto& result_node : body_model->get_results()) {
        ov_outputs.push_back(result_node->input_value(0));
    }
}

void TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model,
                                       std::shared_ptr<ov::Model>& ov_model) {
    OpMap ng_op_map;
    ov::ParameterVector params;
    ov::ResultVector results;
    const auto& model_tf = std::dynamic_pointer_cast<InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_tf, "nullptr for InputModel is given for translation into OV Model");
    const auto& operation_places = model_tf->get_op_places();
    const auto& model_inputs = model_tf->get_inputs();
    const auto& model_outputs = model_tf->get_outputs();
    const auto& model_frozen_inputs = model_tf->get_tensor_values();

    // fill ng_op_map with Constant outputs for frozen inputs
    for (const auto& frozen_input : model_frozen_inputs) {
        const auto& frozen_input_name = frozen_input.first;
        const auto& frozen_input_value = frozen_input.second;
        FRONT_END_GENERAL_CHECK(ng_op_map.count(frozen_input_name) == 0,
                                "Input with frozen value has been already met: " + frozen_input_name);
        ng_op_map[frozen_input_name] = {frozen_input_value};
    }
    // create parameter nodes for all tensor places corresponding to inputs
    for (const auto& input_place : model_inputs) {
        FRONT_END_GENERAL_CHECK(input_place->get_names().size() == 1, "Input place must have one name.");
        auto input_name = input_place->get_names()[0];
        if (ng_op_map.count(input_name)) {
            // probably this input is frozen
            continue;
        }
        const auto& input_tensor_place = std::dynamic_pointer_cast<TensorPlace>(input_place);
        auto input_shape = input_tensor_place->get_partial_shape();
        auto input_type = input_tensor_place->get_element_type();

        // in case of cutting graph, types of custom inputs can be undefined,
        // according to MO help, fp32 is used by default in such cases
        if (input_type == element::undefined) {
            input_type = element::f32;
        }

        auto param = std::make_shared<ov::opset8::Parameter>(input_type, input_shape);
        set_node_name(input_name, param);
        params.push_back(param);
        ng_op_map[input_name] = {param};
    }

    // create the OV ops from TensorFlow ops
    for (const auto& operation_place : operation_places) {
        auto operation_decoder = operation_place->get_decoder();
        auto operation_name = operation_place->get_names()[0];
        // output for parameter nodes has been already generated
        if (ng_op_map.count(operation_name)) {
            continue;
        }

        // prepare a list of OV node inputs for each node
        ov::OutputVector ov_inputs;
        size_t operation_input_size = operation_decoder->get_input_size();

        if (operation_decoder->get_op_type() == "NextIteration") {
            // we expect no inputs for NextIteration because we break-up the cycle in InputModel
            operation_input_size = 0;
        }
        for (size_t input_port_idx = 0; input_port_idx < operation_input_size; ++input_port_idx) {
            // TODO: Implement more general approach. Skipping Constants that have input edges
            if (operation_decoder->get_op_type() == "Const") {
                break;
            }
            std::string producer_name;
            size_t producer_port_idx;
            try {
                operation_decoder->get_input_node(input_port_idx, producer_name, producer_port_idx);
            } catch (const std::exception&) {
                FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(input_port_idx) +
                                " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                producer_name + "', expected input port index: " + std::to_string(producer_port_idx) +
                                '\n');
            }

            // skip conditional edges that must be resolved before operation translation
            // now we can meet them because we still work with TensorFlow protobuf
            if (is_conditional_edge(producer_name)) {
                continue;
            }

            // TODO: re-implement the logic below once Place graph structure is implemented
            // Using Place graph structure (OpPlace, In/OutPortPlace places and their connections) can give
            // names of ports and operations that can be used for further check about existence in ng_op_map

            // check if output vector for places have been already defined and the order of this check is important
            // it moves from places corresponding to input port of the current operation node to output port of original
            // producers
            if (ng_op_map.count(std::to_string(input_port_idx) + ":" + operation_name)) {
                const auto& input_outputs_vector = ng_op_map.at(std::to_string(input_port_idx) + ":" + operation_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name + ":" + std::to_string(producer_port_idx))) {
                const auto& input_outputs_vector =
                    ng_op_map.at(producer_name + ":" + std::to_string(producer_port_idx));
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() == 1,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(0));
            } else if (ng_op_map.count(producer_name)) {
                const auto& input_outputs_vector = ng_op_map.at(producer_name);
                FRONT_END_GENERAL_CHECK(input_outputs_vector.size() > producer_port_idx,
                                        "Input created with pruning must have one output");
                ov_inputs.push_back(input_outputs_vector.at(producer_port_idx));
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "No input is found for node \"" + operation_name + "\" by port " +
                                            std::to_string(producer_port_idx));
            }
        }

        // generate OV node output vector for the current operation node
        ov::OutputVector ov_outputs;
        bool is_converted = false;
        auto operation_type = operation_decoder->get_op_type();
        try {
            if (m_translator_map->count(operation_type)) {
                auto translator = m_translator_map->at(operation_decoder->get_op_type());
                NodeContext node_context(this, operation_decoder, ov_inputs);
                ov_outputs = translator(node_context);
                is_converted = true;
            } else if (auto body_ov_model = get_body_ov_model(operation_type)) {
                inject_body_model(body_ov_model, operation_type, ov_inputs, ov_outputs);
                is_converted = true;
            }
            FRONT_END_OP_CONVERSION_CHECK(is_converted, "No translator found for " + operation_type + " node.");
        } catch (...) {
            if (m_failed_fast) {
                // in case of decode, unsupported operation will be converted to FrameworkNode
                if (m_telemetry && !is_converted) {
                    // send event about which operation is not supported for conversion
                    m_telemetry_data->push_back(
                        std::make_pair<std::string, std::string>("error_cause", "tf_" + operation_type));
                }
                // re-throw any exception
                throw;
            } else {
                auto ng_node = std::make_shared<FrameworkNode>(operation_decoder,
                                                               ov_inputs,
                                                               operation_place->get_output_ports().size());
                set_node_name(operation_name, ng_node);
                ov_outputs = ng_node->outputs();
            }
        }

        // register OV node outputs in the map for new operation node
        for (const auto& output : ov_outputs) {
            if (auto result = std::dynamic_pointer_cast<ov::opset10::Result>(output.get_node_shared_ptr())) {
                // do not add RetVal type operation to ng_op_map
                results.push_back(result);
            } else {
                auto param = std::dynamic_pointer_cast<ov::opset8::Parameter>(output.get_node_shared_ptr());
                // avoid duplicating Parameter nodes if they are already in the Parameters vector
                if (param && operation_decoder->get_op_type() != "Identity" &&
                    std::find(params.begin(), params.end(), param) == params.end()) {
                    params.push_back(param);
                }
                ng_op_map[operation_name].push_back(output);
            }
        }
    }

    // create Result nodes for all model outputs
    if (results.empty()) {
        for (const auto& model_output : model_outputs) {
            auto model_output_tensor_place = std::dynamic_pointer_cast<TensorPlace>(model_output);
            auto model_output_name = model_output_tensor_place->get_names()[0];
            std::string operation_name;
            std::string port_type;
            size_t port_index;
            ov::frontend::tensorflow::extract_operation_name_and_port(model_output_name,
                                                                      operation_name,
                                                                      port_index,
                                                                      port_type);

            if (port_type == "none") {
                for (const auto& node_output : ng_op_map[operation_name]) {
                    auto result_node = std::make_shared<ov::opset8::Result>(node_output);
                    result_node->set_friendly_name(model_output_name);
                    results.push_back(result_node);
                }
            } else if (port_type == "out") {
                const auto& node_outputs = ng_op_map[operation_name];
                FRONT_END_GENERAL_CHECK(node_outputs.size() > port_index,
                                        "Output port with index " + std::to_string(port_index) + " of " +
                                            operation_name + "node specified as custom output does not exist");
                auto result_node = std::make_shared<ov::opset8::Result>(node_outputs[port_index]);
                result_node->set_friendly_name(model_output_name);
                results.push_back(result_node);
            } else if (port_type == "in") {
                // TODO: avoid this traversing by having a map for OpPlace objects, for example
                std::shared_ptr<OpPlace> operation_place = nullptr;
                for (const auto& op_place : operation_places) {
                    FRONT_END_GENERAL_CHECK(!op_place->get_names().empty(), "No names for OpPlace found.");
                    if (op_place->get_names()[0] == operation_name) {
                        operation_place = op_place;
                    }
                }
                FRONT_END_GENERAL_CHECK(operation_place, "There is no operation place with a name: " + operation_name);
                auto operation_decoder = operation_place->get_decoder();

                // get to know a producer node and by which its output port data is generated
                std::string producer_name;
                size_t producer_port_idx;
                try {
                    operation_decoder->get_input_node(port_index, producer_name, producer_port_idx);
                } catch (const std::exception&) {
                    FRONT_END_THROW("[ ERROR ] Exception happened when preparing input " + std::to_string(port_index) +
                                    " for op '" + operation_decoder->get_op_name() + "', expected input name: '" +
                                    producer_name +
                                    "', expected input port index: " + std::to_string(producer_port_idx) + '\n');
                }

                // add Result node for this producer output port
                const auto& node_outputs = ng_op_map[producer_name];
                FRONT_END_GENERAL_CHECK(node_outputs.size() > producer_port_idx,
                                        "Output port with index " + std::to_string(producer_port_idx) + " of " +
                                            producer_name + "node specified as custom output does not exist");
                auto result_node = std::make_shared<ov::opset8::Result>(node_outputs[producer_port_idx]);
                result_node->set_friendly_name(model_output_name);
                results.push_back(result_node);
            }
        }
    }

    // TODO: it may be redundant step since models_output is filled in InputModel constructor
    // find all terminal nodes in OV graph to complete list of results
    if (results.empty()) {
        for (const auto& node_output_vector : ng_op_map) {
            for (size_t output_ind = 0; output_ind < node_output_vector.second.size(); ++output_ind) {
                auto output = node_output_vector.second[output_ind];
                if (output.get_target_inputs().empty() &&
                    !std::dynamic_pointer_cast<ov::opset8::Result>(output.get_node_shared_ptr())) {
                    auto model_output_name =
                        output.get_node_shared_ptr()->get_friendly_name() + ":" + std::to_string(output_ind);
                    auto result_node = std::make_shared<ov::opset8::Result>(output);
                    result_node->set_friendly_name(model_output_name);
                    results.push_back(result_node);
                }
            }
        }
    }

    // TODO: reorder results and params according to indices given in RT info (if any)

    // create the OV Model
    ov_model = std::make_shared<ov::Model>(results, params, m_model_name);
}

std::shared_ptr<ov::Model> TranslateSession::get_body_ov_model(const std::string& body_graph_name) {
    std::shared_ptr<ov::Model> body_model = nullptr;
    auto input_model = std::dynamic_pointer_cast<InputModel>(m_input_model);
    if (m_cached_body_models->count(body_graph_name)) {
        // check if such body graph has been converted before
        // re-use it from the cache for further injection

        // create new instance of the required body model
        // since it will be modified by injection
        auto cached_body_model = m_cached_body_models->at(body_graph_name);
        body_model = cached_body_model->clone();
    } else if (auto body_input_model = input_model->get_body_input_model(body_graph_name)) {
        // try to find a function by name in the model library
        translate_graph(body_input_model, body_model);
        // save new instance of body_model in the cache of body models
        // before its injection into the parent graph

        auto cached_body_model = body_model->clone();
        update_cached_body_models(body_graph_name, cached_body_model);
    }
    return body_model;
}
