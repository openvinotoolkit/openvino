// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "core/null_node.hpp"
#include "input_model.hpp"
#include "onnx_framework_node.hpp"
#include "openvino/frontend/onnx/decoder.hpp"
#include "openvino/frontend/onnx/graph_iterator.hpp"
#include "openvino/op/util/op_types.hpp"
#include "ops_bridge.hpp"
#include "place.hpp"

using namespace ov::frontend::onnx;

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::shared_ptr<OperatorsBridge>& translator_map,
                                   const std::string& model_name)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_model_name(model_name),
      m_ov_model(nullptr),
      m_fail_fast(false),
      m_parent_session(nullptr) {}

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   TranslateSession* parent_session,
                                   const std::string& model_name)
    : m_input_model(input_model),
      m_translator_map(parent_session->m_translator_map),
      m_model_name(model_name),
      m_ov_model(nullptr),
      m_fail_fast(false),
      m_parent_session(parent_session) {}

ov::Output<ov::Node> TranslateSession::lookup_tensor(const std::string& name) {
    auto local_tensor = m_tensor_values.find(name);
    if (local_tensor != m_tensor_values.end()) {
        return local_tensor->second;
    }
    if (m_parent_session != nullptr) {
        auto node_from_parent = m_parent_session->lookup_tensor(name);
        if (node_from_parent.get_node() == nullptr) {
            return {};
        }
        if (ov::op::util::is_constant(node_from_parent.get_node_shared_ptr())) {
            return node_from_parent;
        }
        auto new_param = std::make_shared<ov::op::v0::Parameter>(node_from_parent.get_element_type(),
                                                                 node_from_parent.get_partial_shape());
        new_param->set_friendly_name(node_from_parent.get_node()->get_friendly_name());
        m_parameters.push_back(new_param);
        m_tensor_values[name] = new_param;
        return new_param;
    }
    return {};
}

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
    const auto model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);

    auto& all_tensor_places = model_onnx->get_tensor_places();

    // inputs
    m_parameters.reserve(model_onnx->get_inputs().size());

    // Lambda detects type of input_tensor and creates correct node: constant or parameter
    auto create_const_or_param = [&](const std::string& name,
                                     const std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>& input_tensor) {
        std::shared_ptr<ov::Node> node;
        if (input_tensor->get_data_location() != nullptr || input_tensor->get_data() != nullptr) {
            Tensor tensor = Tensor(input_tensor);
            node = tensor.get_ov_constant();
        } else if (input_tensor->get_partial_shape() == PartialShape{0}) {  // empty constant
            node = ov::op::v0::Constant::create(input_tensor->get_element_type(),
                                                input_tensor->get_partial_shape().to_shape(),
                                                {});
        } else {
            node = std::make_shared<ov::op::v0::Parameter>(input_tensor->get_element_type(),
                                                           input_tensor->get_partial_shape());
            m_parameters.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(node));
        }
        node->set_friendly_name(name);
        m_tensor_values[name] = node->get_default_output();
        input_tensor->translate(m_tensor_values[name]);
    };

    for (const auto& input : model_onnx->get_inputs()) {
        const auto input_tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(input);
        FRONT_END_GENERAL_CHECK(input_tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorONNXPlace instances");
        const auto name = input_tensor->get_names()[0];
        create_const_or_param(name, input_tensor);
    }

    // operations
    for (const auto& op_place : model_onnx->get_op_places()) {
        const auto decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(op_place->get_decoder());
        FRONT_END_GENERAL_CHECK(decoder != nullptr, "Decoder must be onnx::DecoderBase or its child");
        for (size_t i = 0; i < decoder->get_input_size(); ++i) {
            const auto& name = decoder->get_input_tensor_name(i);
            if (name == "") {
                continue;
            }
            auto node = lookup_tensor(name);
            if (node.get_node() == nullptr) {
                auto place_it = all_tensor_places.find(name);
                FRONT_END_GENERAL_CHECK(place_it != all_tensor_places.end(), "Tensor place not found in a graph");
                create_const_or_param(name, place_it->second);
            }
        }

        const auto out_size = decoder->get_output_size();
        ov::OutputVector ov_outputs(out_size);
        const Operator* translator =
            translate_map.get_operator(decoder->get_domain(), decoder->get_op_type(), decoder->get_op_set());
        ov::frontend::onnx::Node node_context(*decoder, this);
        std::string error_message{};
        try {
            if (translator == nullptr) {
                ov_outputs = std::make_shared<ov::frontend::onnx::ONNXFrameworkNode>(node_context)->outputs();
            } else {
                ov_outputs = (*translator)(node_context);
            }
            for (size_t idx = 0; idx < ov_outputs.size() && idx < out_size; ++idx) {
                const std::string& out_name = decoder->get_output_tensor_name(idx);
                if (is_optimized_out(ov_outputs[idx])) {
                    ov_outputs[idx].add_names({out_name});
                } else {
                    ov_outputs[idx].set_names({out_name});
                    ov_outputs[idx].get_node()->set_friendly_name(out_name);
                }
            }
        } catch (const ::ov::frontend::onnx::error::OnnxNodeValidationFailure& e) {
            error_message = e.what();
        } catch (const std::exception& exc) {
            error_message = error::detail::get_error_msg_prefix(node_context);
            error_message += ": " + std::string{exc.what()};
        } catch (...) {
            error_message = error::detail::get_error_msg_prefix(node_context);
            // Since we do not know anything about current exception data type we can only
            // notify user in this way.
            error_message += "Unhandled exception type. \n";
        }
        if (!error_message.empty()) {
            auto telemetry = model_onnx->get_telemetry_extension();
            if (m_fail_fast) {
                if (telemetry && translator == nullptr) {
                    telemetry->send_event("error_cause", "onnx_" + decoder->get_op_type());
                }
                throw;
            } else {
                if (telemetry && !error_message.empty()) {
                    std::string onnx_domain = decoder->get_domain();
                    uint64_t opset_version = decoder->get_op_set();
                    error_message = "[ONNX Frontend] Conversion failed for " +
                                    (onnx_domain != "" ? "***." + decoder->get_op_type() + "-X"
                                                       : decoder->get_op_type() + "-" + std::to_string(opset_version)) +
                                    "\n" + error_message;
                }
                auto operation =
                    std::make_shared<ov::frontend::onnx::NotSupportedONNXNode>(node_context.get_ov_inputs(),
                                                                               decoder->get_output_size(),
                                                                               decoder->get_domain(),
                                                                               decoder->get_op_type(),
                                                                               error_message);
                operation->set_friendly_name(decoder->get_op_name());
                ov_outputs = operation->outputs();
            }
        }
        for (size_t i = 0; i < ov_outputs.size() && i < decoder->get_output_size(); ++i) {
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
        const auto tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output);
        FRONT_END_GENERAL_CHECK(tensor != nullptr,
                                "Inputs of ov::frontend::onnx::InputModel must be TensorLitePlace instances");
        const auto name = tensor->get_names()[0];
        if (!m_tensor_values.count(name)) {
            continue;
        }
        const auto& output_value = m_tensor_values[name];
        const auto result = std::make_shared<ov::op::v0::Result>(output_value);
        auto input = result->output(0);
        tensor->translate(input);
        result->set_friendly_name(name + "/sink_port_0");
        results.push_back(result);
    }

    auto model_name = "onnx_Frontend_IR";
    ov_model = std::make_shared<ov::Model>(results, m_parameters, model_name);
}