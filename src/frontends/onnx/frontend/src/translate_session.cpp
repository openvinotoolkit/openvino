// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "core/null_node.hpp"
#include "input_model.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "ops_bridge.hpp"
#include "place.hpp"

using namespace ov::frontend::onnx;

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::shared_ptr<OperatorsBridge>& translator_map,
                                   const std::string& model_name)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_model_name(model_name),
      m_ov_model(nullptr) {}

std::shared_ptr<ov::Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    translate_graph(m_input_model, m_ov_model);
    return m_ov_model;
}

void TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model,
                                       std::shared_ptr<ov::Model>& ov_model) {
    const OperatorsBridge translate_map;
    const auto& model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);

    auto& all_tensor_places = model_onnx->get_tensor_places();

    std::map<std::string, Output<ov::Node>> parent_tensors;

    // inputs
    ParameterVector parameters;
    parameters.reserve(model_onnx->get_inputs().size());
    for (const auto& input : model_onnx->get_inputs()) {
        const auto& input_tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(input);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorONNXPlace instances");
        const auto name = input_tensor->get_names()[0];
        auto parameter = std::make_shared<ov::op::v0::Parameter>(input_tensor->get_element_type(),
                                                                 input_tensor->get_partial_shape());
        parameter->set_friendly_name(name);
        parameters.push_back(parameter);
        // Do not overwrite an existing tensors
        // Usually it means a parent graph already has a node with a same tensor name
        // But an order for a further lookup tensors: try to find a local Parameter, and only after that -
        // request for a known tensor of above layer
        if (m_tensor_values.count(name) > 0) {
            parent_tensors[name] = m_tensor_values[name];
        }
        m_tensor_values[name] = parameter->get_default_output();
        input_tensor->translate(m_tensor_values[name]);
    }

    // operations
    for (const auto& op_place : model_onnx->get_op_places()) {
        const auto& decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(op_place->get_decoder());
        FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder must be onnx::DecoderBase or its child");
        for (size_t i = 0; i < decoder->get_input_size(); ++i) {
            const auto& name = decoder->get_input_tensor_name(i);
            if (name == "") {
                continue;
            }
            auto tensor_it = m_tensor_values.find(name);
            // If tensor wasn't found - probably we may need to find it another way
            if (tensor_it == m_tensor_values.end() ||
                std::dynamic_pointer_cast<ov::op::v0::Parameter>(tensor_it->second.get_node_shared_ptr())) {
                auto place_it = all_tensor_places.find(name);
                if (place_it != all_tensor_places.end()) {
                    if (auto data = place_it->second->get_data()) {
                        auto constant = ov::op::v0::Constant::create(place_it->second->get_element_type(),
                                                                     place_it->second->get_partial_shape().to_shape(),
                                                                     data);
                        constant->set_friendly_name(place_it->first);
                        m_tensor_values[place_it->first] = constant;
                        continue;
                    } else if (place_it->second->get_partial_shape() == PartialShape{0}) {  // empty constant
                        auto constant = ov::op::v0::Constant::create(place_it->second->get_element_type(),
                                                                     place_it->second->get_partial_shape().to_shape(),
                                                                     {});
                        constant->set_friendly_name(place_it->first);
                        m_tensor_values[place_it->first] = constant;
                        continue;
                    }
                }
            }
        }

        const auto& out_size = decoder->get_output_size();
        ov::OutputVector ov_outputs(out_size);
        const Operator* translator =
            translate_map.get_operator(decoder->get_domain(), decoder->get_op_type(), decoder->get_op_set());
        try {
            FRONT_END_OP_CONVERSION_CHECK(
                translator != nullptr,
                "No translator found for " + decoder->get_domain() + " " + decoder->get_op_type() + " node.");
            const NodeProto* node_def = nullptr;
            decoder->experimental_get_internal_structures(reinterpret_cast<const void**>(&node_def));
            ov::frontend::onnx::Node node_context(*decoder, this);
            ov_outputs = (*translator)(node_context);
            for (size_t idx = 0; idx < ov_outputs.size(); ++idx) {
                const std::string& out_name = decoder->get_output_tensor_name(idx);
                ov_outputs[idx].set_names({out_name});
                ov_outputs[idx].get_node()->set_friendly_name(out_name);
            }
        } catch (...) {
            /*
            if (fail_fast) {
                if (m_telemetry && translator == nullptr) {
                    m_telemetry->send_event("error_cause", "onnx_" + decoder->get_op_type());
                }
                throw;
            } else
            {
                auto operation = std::make_shared<ov::frontend::onnx::FrameworkNode>(decoder, inputs, out_size);
                operation->set_friendly_name(decoder->get_op_name());
                ov_outputs = operation->outputs();
            }
            */
        }
        for (size_t i = 0; i < ov_outputs.size(); ++i) {
            const auto& name = decoder->get_output_tensor_name(i);
            if (name == "") {
                // Means - not connected
                continue;
            }
            m_tensor_values[name] = ov_outputs[i];
            all_tensor_places[name]->translate(m_tensor_values[name]);
        }
    }

    // outputs
    ResultVector results;
    results.reserve(model_onnx->get_outputs().size());
    for (const auto& output : model_onnx->get_outputs()) {
        const auto& tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output);
        FRONT_END_GENERAL_CHECK(tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorLitePlace instances");
        const auto name = tensor->get_names()[0];
        if (!m_tensor_values.count(name)) {
            continue;
        }
        const auto& output_value = m_tensor_values[name];
        const auto& result = std::make_shared<ov::op::v0::Result>(output_value);
        auto input = result->output(0);
        tensor->translate(input);
        result->set_friendly_name(name + "/sink_port_0");
        results.push_back(result);
    }

    // Restoring links on a parent tensors
    for (auto& parent_tensor : parent_tensors) {
        m_tensor_values[parent_tensor.first] = parent_tensor.second;
    }

    auto model_name = "onnx_Frontend_IR";
    ov_model = std::make_shared<ov::Model>(results, parameters, model_name);
}