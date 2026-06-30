// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "core/null_node.hpp"
#include "core/tensor.hpp"
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
        new_param->output(0).set_names({name});
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
    const auto model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");

    if (!model_onnx->is_loaded()) {
        // Model is not loaded to Place-graph
        // use iterator directly to convert it to ov::Model.
        translate_graph_from_iterator(input_model, ov_model);
        return;
    }

    auto& all_tensor_places = model_onnx->get_tensor_places();

    // inputs
    m_parameters.reserve(model_onnx->get_inputs().size());

    // Lambda detects type of input_tensor and creates correct node: constant or parameter
    auto create_const_or_param = [&](const std::string& name,
                                     const std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>& input_tensor) {
        std::shared_ptr<ov::Node> node;
        // STRING initializers carry their data in get_data_any() with get_data()==nullptr; treat
        // them as Constants too.
        if (input_tensor->get_data_location() != nullptr || input_tensor->get_data() != nullptr ||
            !input_tensor->get_data_any().empty()) {
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
            m_translator_map->get_operator(decoder->get_domain(), decoder->get_op_type(), decoder->get_op_set());
        ov::frontend::onnx::Node node_context(decoder, this);
        std::string error_message{};
        try {
            if (translator == nullptr) {
                ov_outputs = std::make_shared<ov::frontend::onnx::ONNXFrameworkNode>(node_context)->outputs();
            } else {
                ov_outputs = (*translator)(node_context);
            }
            auto name = node_context.get_name();
            for (size_t idx = 0; idx < ov_outputs.size() && idx < out_size; ++idx) {
                const std::string& out_name = node_context.output(static_cast<int>(idx));
                if (is_optimized_out(ov_outputs[idx])) {
                    ov_outputs[idx].add_names({out_name});
                } else {
                    ov_outputs[idx].set_names({out_name});
                    ov_outputs[idx].get_node()->set_friendly_name(!name.empty() ? name : out_name);
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
            std::string onnx_domain = decoder->get_domain();
            uint64_t opset_version = decoder->get_op_set();
            if (m_fail_fast) {
                if (telemetry && translator == nullptr) {
                    telemetry->send_event("error_cause", "onnx_" + decoder->get_op_type());
                }
                FRONT_END_THROW(error_message);
            } else {
                if (telemetry && !error_message.empty()) {
                    error_message = "[ONNX Frontend] Conversion failed for " +
                                    (onnx_domain != "" ? "***." + decoder->get_op_type() + "-X"
                                                       : decoder->get_op_type() + "-" + std::to_string(opset_version)) +
                                    "\n" + error_message;
                }
                auto operation =
                    std::make_shared<ov::frontend::onnx::NotSupportedONNXNode>(node_context.get_ov_inputs(),
                                                                               decoder->get_output_size(),
                                                                               onnx_domain,
                                                                               decoder->get_op_type(),
                                                                               static_cast<int64_t>(opset_version),
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
    // Materialize any output tensors that are direct constants (initializers used as graph
    // outputs without being consumed by any op).  These are not created during the inputs or
    // operations loops because they have data but are not referenced as op inputs.
    // Also handle subgraph outputs that reference parent scope tensors — lookup_tensor()
    // will create a Parameter for any parent-scope non-constant value.
    ResultVector results;
    results.reserve(model_onnx->get_outputs().size());
    for (const auto& output : model_onnx->get_outputs()) {
        const auto tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output);
        FRONT_END_GENERAL_CHECK(tensor != nullptr,
                                "Outputs of ov::frontend::onnx::InputModel must be TensorONNXPlace instances");
        const auto name = tensor->get_names()[0];
        if (!m_tensor_values.count(name)) {
            auto place_it = all_tensor_places.find(name);
            if (place_it != all_tensor_places.end() &&
                (place_it->second->get_data() != nullptr || place_it->second->get_data_location() != nullptr)) {
                create_const_or_param(name, place_it->second);
            } else if (auto parent_value = lookup_tensor(name); parent_value.get_node() != nullptr) {
                // lookup_tensor() resolved the name from a parent scope. For non-constant
                // parent values it already cached a Parameter in m_tensor_values; for
                // parent-scope Constants it returns the Constant directly without caching,
                // so insert it here to make the subsequent m_tensor_values[name] lookup
                // safe.
                m_tensor_values.emplace(name, parent_value);
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "Output tensor \"",
                                        name,
                                        "\" was declared as a graph output but is not produced by any operation, "
                                        "is not an initializer, and cannot be resolved from a parent scope.");
            }
        }
        const auto& output_value = m_tensor_values[name];
        const auto result = std::make_shared<ov::op::v0::Result>(output_value);
        auto input = result->output(0);
        tensor->translate(input);
        result->set_friendly_name(name + "/sink_port_0");
        results.push_back(result);
        const auto& previous_operation = result->get_input_node_shared_ptr(0);
        if (!ov::as_type_ptr<ov::op::v0::Parameter>(previous_operation)) {
            previous_operation->set_friendly_name(name);
        }
    }

    auto model_name = "onnx_Frontend_IR";
    ov_model = std::make_shared<ov::Model>(results, m_parameters, model_name);

    const auto& metadata = model_onnx->get_metadata();
    const std::string framework_section = "framework";
    for (const auto& pair : metadata) {
        ov_model->set_rt_info(pair.second, framework_section, pair.first);
    }
}

namespace {
// Materialize a TensorONNXPlace from a TensorMetaInfo without registering it in any Place map.
// Mirrors decode_tensor_place() in input_model.cpp; used only to feed Tensor::get_ov_constant().
std::shared_ptr<ov::frontend::onnx::TensorONNXPlace> make_transient_tensor_place(
    const ov::frontend::onnx::TensorMetaInfo& info,
    const ov::frontend::InputModel& model,
    const bool reuse_const_data) {
    return std::make_shared<ov::frontend::onnx::TensorONNXPlace>(
        model,
        info.m_partial_shape,
        info.m_element_type,
        std::vector<std::string>{info.m_tensor_name ? *info.m_tensor_name : std::string{}},
        info.m_tensor_data,
        info.m_tensor_data_size,
        info.m_tensor_data_any,
        info.m_external_location,
        info.m_is_raw,
        reuse_const_data);
}

// A tensor is a Constant iff it carries inline data, points at external data, or holds STRING data
// (stored in m_tensor_data_any with m_tensor_data == nullptr).
bool tensor_has_data(const ov::frontend::onnx::TensorMetaInfo& info) {
    return info.m_tensor_data != nullptr || info.m_external_location != nullptr || !info.m_tensor_data_any.empty();
}
}  // namespace

void TranslateSession::translate_graph_from_iterator(const ov::frontend::InputModel::Ptr& input_model,
                                                     std::shared_ptr<ov::Model>& ov_model) {
    const auto model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");

    const auto graph_iterator =
        std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(model_onnx->get_graph_iterator());
    FRONT_END_GENERAL_CHECK(graph_iterator != nullptr, "Invalid graph iterator for single-pass conversion");

    const auto telemetry = model_onnx->get_telemetry_extension();
    // Preserve zero-copy constant wrapping when the iterator's owner allows it (e.g. an EP delegate
    // passing reuse_const_data=true); the two-pass path carries this through the Place graph.
    const bool reuse_const_data = model_onnx->is_const_data_reusable();

    // A data-carrying tensor becomes a Constant via a transient TensorONNXPlace (registered in no map)
    // fed to Tensor::get_ov_constant() — the same materialization the two-pass path uses.
    const auto make_constant = [&](const ov::frontend::onnx::TensorMetaInfo& info) -> std::shared_ptr<ov::Node> {
        auto place = make_transient_tensor_place(info, *model_onnx, reuse_const_data);
        return ov::frontend::onnx::Tensor(place).get_ov_constant();
    };

    static const std::string empty_tensor_name;

    // Materialize a Constant (data) or Parameter (no data) from a tensor's meta info and record it in
    // m_tensor_values. The caller decides whether a no-data node is a graph input (m_parameters).
    auto create_const_or_param = [&](const std::string& name,
                                     const ov::frontend::onnx::TensorMetaInfo& info) -> std::shared_ptr<ov::Node> {
        std::shared_ptr<ov::Node> node;
        if (tensor_has_data(info)) {
            node = make_constant(info);
        } else if (info.m_partial_shape == PartialShape{0}) {  // empty constant
            node = ov::op::v0::Constant::create(info.m_element_type, info.m_partial_shape.to_shape(), {});
        } else {
            node = std::make_shared<ov::op::v0::Parameter>(info.m_element_type, info.m_partial_shape);
        }
        node->set_friendly_name(name);
        auto out = node->get_default_output();
        // Mirror TensorONNXPlace::translate(): the ONNX tensor name must be in the output's name set,
        // since input/output identification depends on it.
        if (!name.empty()) {
            out.add_names({name});
        }
        m_tensor_values[name] = out;
        return node;
    };

    // Resolve an op input by name: reuse an already-produced value, else materialize a Constant/Parameter
    // from the decoder's input tensor info. Falls back to lookup_tensor (parent scope) on a miss.
    auto resolve_input = [&](const std::string& name, const ov::frontend::onnx::TensorMetaInfo& info) {
        if (m_tensor_values.find(name) != m_tensor_values.end()) {
            return;
        }
        if (lookup_tensor(name).get_node() != nullptr) {
            return;
        }
        create_const_or_param(name, info);
    };

    // Graph inputs/outputs with their model-defined index for stable ordering. Outputs hold the owning
    // tensor decoder (not a raw TensorMetaInfo*): a lazy GraphIterator may free a decoder once next()
    // advances past it, so keeping the shared_ptr alive avoids a use-after-free when get_tensor_info()
    // is re-fetched after the loop for a pass-through initializer output.
    std::vector<std::pair<int64_t, std::shared_ptr<ov::op::v0::Parameter>>> indexed_inputs;
    std::vector<std::tuple<int64_t, std::string, std::shared_ptr<onnx::DecoderBaseTensor>>> indexed_outputs;

    std::map<std::string, uint64_t> op_statistics;  // for telemetry

    for (; !graph_iterator->is_end(); graph_iterator->next()) {
        const auto& decoder = graph_iterator->get_decoder();

        if (const auto tensor_decoder = std::dynamic_pointer_cast<onnx::DecoderBaseTensor>(decoder)) {
            const auto& info = tensor_decoder->get_tensor_info();
            const auto input_idx = tensor_decoder->get_input_idx();
            const auto output_idx = tensor_decoder->get_output_idx();
            const bool has_data = tensor_has_data(info);

            // Pure constant/initializer (not a graph output): materialized on demand by ops.
            if (has_data && output_idx < 0) {
                continue;
            }
            const std::string& name = info.m_tensor_name ? *info.m_tensor_name : empty_tensor_name;
            if (input_idx >= 0 && !has_data) {
                auto node = create_const_or_param(name, info);
                // A no-data graph input is a Parameter unless it is the empty-constant special case.
                if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
                    indexed_inputs.emplace_back(input_idx, param);
                }
            }
            if (output_idx >= 0) {
                indexed_outputs.emplace_back(output_idx, name, tensor_decoder);
            }
            continue;
        }

        const auto op_decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(decoder);
        FRONT_END_GENERAL_CHECK(op_decoder != nullptr, "Decoder must be onnx::DecoderBaseOperation or its child");

        // Ensure all inputs exist (intermediate values were produced by earlier ops in topo order).
        for (size_t i = 0; i < op_decoder->get_input_size(); ++i) {
            const auto& name = op_decoder->get_input_tensor_name(i);
            if (name.empty()) {
                continue;
            }
            resolve_input(name, op_decoder->get_input_tensor_info(i));
        }

        const auto out_size = op_decoder->get_output_size();
        ov::OutputVector ov_outputs(out_size);
        const Operator* translator = m_translator_map->get_operator(op_decoder->get_domain(),
                                                                    op_decoder->get_op_type(),
                                                                    op_decoder->get_op_set());
        ov::frontend::onnx::Node node_context(op_decoder, this);
        std::string error_message{};

        if (telemetry) {
            // Key by the iterator's resolved opset (matches load_model()'s telemetry) so op_count
            // events are identical whichever conversion path runs.
            op_statistics[op_decoder->get_op_type() + "-" +
                          std::to_string(graph_iterator->get_opset_version(op_decoder->get_domain()))]++;
        }

        try {
            if (translator == nullptr) {
                ov_outputs = std::make_shared<ov::frontend::onnx::ONNXFrameworkNode>(node_context)->outputs();
            } else {
                ov_outputs = (*translator)(node_context);
            }
            auto name = node_context.get_name();
            for (size_t idx = 0; idx < ov_outputs.size() && idx < out_size; ++idx) {
                const std::string& out_name = node_context.output(static_cast<int>(idx));
                if (is_optimized_out(ov_outputs[idx])) {
                    ov_outputs[idx].add_names({out_name});
                } else {
                    ov_outputs[idx].set_names({out_name});
                    ov_outputs[idx].get_node()->set_friendly_name(!name.empty() ? name : out_name);
                }
            }
        } catch (const ::ov::frontend::onnx::error::OnnxNodeValidationFailure& e) {
            error_message = e.what();
        } catch (const std::exception& exc) {
            error_message = error::detail::get_error_msg_prefix(node_context);
            error_message += ": " + std::string{exc.what()};
        } catch (...) {
            error_message = error::detail::get_error_msg_prefix(node_context);
            error_message += "Unhandled exception type. \n";
        }
        if (!error_message.empty()) {
            std::string onnx_domain = op_decoder->get_domain();
            uint64_t opset_version = op_decoder->get_op_set();
            if (m_fail_fast) {
                if (telemetry && translator == nullptr) {
                    telemetry->send_event("error_cause", "onnx_" + op_decoder->get_op_type());
                }
                FRONT_END_THROW(error_message);
            } else {
                if (telemetry) {
                    error_message =
                        "[ONNX Frontend] Conversion failed for " +
                        (onnx_domain != "" ? "***." + op_decoder->get_op_type() + "-X"
                                           : op_decoder->get_op_type() + "-" + std::to_string(opset_version)) +
                        "\n" + error_message;
                }
                auto operation =
                    std::make_shared<ov::frontend::onnx::NotSupportedONNXNode>(node_context.get_ov_inputs(),
                                                                               op_decoder->get_output_size(),
                                                                               onnx_domain,
                                                                               op_decoder->get_op_type(),
                                                                               static_cast<int64_t>(opset_version),
                                                                               error_message);
                operation->set_friendly_name(op_decoder->get_op_name());
                ov_outputs = operation->outputs();
            }
        }
        for (size_t i = 0; i < ov_outputs.size() && i < out_size; ++i) {
            const auto& name = op_decoder->get_output_tensor_name(i);
            if (name.empty()) {
                continue;
            }
            m_tensor_values[name] = ov_outputs[i];
        }
    }

    if (telemetry) {
        for (const auto& op : op_statistics) {
            telemetry->send_event("op_count", "onnx_" + op.first, static_cast<int>(op.second));
        }
    }

    // Order graph-input parameters by model input index and place them ahead of any parent-scope
    // parameters that lookup_tensor() appended during op translation (subgraph case).
    std::sort(indexed_inputs.begin(), indexed_inputs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    ParameterVector ordered_inputs;
    ordered_inputs.reserve(indexed_inputs.size());
    for (const auto& entry : indexed_inputs) {
        ordered_inputs.push_back(entry.second);
    }
    m_parameters.insert(m_parameters.begin(), ordered_inputs.begin(), ordered_inputs.end());

    // Build results ordered by model output index.
    std::stable_sort(indexed_outputs.begin(), indexed_outputs.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
    ResultVector results;
    results.reserve(indexed_outputs.size());
    for (const auto& entry : indexed_outputs) {
        const auto& name = std::get<1>(entry);
        // Re-fetch from the still-alive decoder; the loop may have advanced (and freed) other decoders.
        const auto& info = std::get<2>(entry)->get_tensor_info();
        if (!m_tensor_values.count(name)) {
            // Output is a direct initializer-as-graph-output, or resolvable from a parent scope.
            if (tensor_has_data(info)) {
                create_const_or_param(name, info);
            } else if (auto parent_value = lookup_tensor(name); parent_value.get_node() != nullptr) {
                m_tensor_values.emplace(name, parent_value);
            } else {
                FRONT_END_GENERAL_CHECK(false,
                                        "Output tensor \"",
                                        name,
                                        "\" was declared as a graph output but is not produced by any operation, "
                                        "is not an initializer, and cannot be resolved from a parent scope.");
            }
        }
        const auto& output_value = m_tensor_values[name];
        const auto result = std::make_shared<ov::op::v0::Result>(output_value);
        // The Result's input port must carry the ONNX output tensor name (mirrors tensor->translate()).
        auto result_input = result->output(0);
        if (!name.empty()) {
            result_input.add_names({name});
        }
        result->set_friendly_name(name + "/sink_port_0");
        results.push_back(result);
        const auto& previous_operation = result->get_input_node_shared_ptr(0);
        if (!ov::as_type_ptr<ov::op::v0::Parameter>(previous_operation)) {
            previous_operation->set_friendly_name(name);
        }
    }

    auto model_name = "onnx_Frontend_IR";
    ov_model = std::make_shared<ov::Model>(results, m_parameters, model_name);

    // Read metadata from the iterator directly: InputModel::get_metadata() is only populated by
    // load_model(), which this path skips. The iterator reads metadata_props independently of the
    // Place graph, restoring "framework" rt_info parity without defeating the load-skip optimization.
    const auto metadata = graph_iterator->get_metadata();
    const std::string framework_section = "framework";
    for (const auto& pair : metadata) {
        ov_model->set_rt_info(pair.second, framework_section, pair.first);
    }
}
