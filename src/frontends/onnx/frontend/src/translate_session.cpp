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

ov::OutputVector TranslateSession::apply_op_translator(const std::shared_ptr<DecoderBaseOperation>& decoder,
                                                       const std::shared_ptr<TelemetryExtension>& telemetry) {
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
        // Unknown exception type; notify the user generically.
        error_message += "Unhandled exception type. \n";
    }
    if (!error_message.empty()) {
        std::string onnx_domain = decoder->get_domain();
        uint64_t opset_version = decoder->get_op_set();
        if (m_fail_fast) {
            if (telemetry && translator == nullptr) {
                telemetry->send_event("error_cause", "onnx_" + decoder->get_op_type());
            }
            FRONT_END_THROW(error_message);
        } else {
            if (telemetry) {
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
    return ov_outputs;
}

std::shared_ptr<ov::Node> TranslateSession::create_const_or_param(
    const std::string& name,
    const std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>& input_tensor) {
    std::shared_ptr<ov::Node> node;
    // STRING initializers carry data in get_data_any() with get_data()==nullptr; treat as Constants.
    if (input_tensor->get_data_location() != nullptr || input_tensor->get_data() != nullptr ||
        !input_tensor->get_data_any().empty()) {
        node = Tensor(input_tensor).get_ov_constant();
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
    // Copy the ONNX tensor name into the output's name set; skip empty (unnamed) tensors.
    if (!name.empty()) {
        input_tensor->translate(m_tensor_values[name]);
    }
    return node;
}

void TranslateSession::send_op_count_telemetry(const std::shared_ptr<TelemetryExtension>& telemetry,
                                               const std::map<std::string, uint64_t>& op_statistics) const {
    if (!telemetry) {
        return;
    }
    for (const auto& op : op_statistics) {
        telemetry->send_event("op_count", "onnx_" + op.first, static_cast<int>(op.second));
    }
}

namespace {
// Build a transient TensorONNXPlace from a TensorMetaInfo, registered in no Place map.
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

// A tensor is a Constant iff it carries inline data, external data, or STRING data.
bool tensor_has_data(const ov::frontend::onnx::TensorMetaInfo& info) {
    return info.m_tensor_data != nullptr || info.m_external_location != nullptr || !info.m_tensor_data_any.empty();
}

// Same predicate as tensor_has_data(), expressed over a materialized TensorONNXPlace.
bool place_has_data(const ov::frontend::onnx::TensorONNXPlace& place) {
    return place.get_data() != nullptr || place.get_data_location() != nullptr || !place.get_data_any().empty();
}

struct DiscoveredOutput {
    int64_t model_output_index;
    std::string tensor_name;
    std::shared_ptr<ov::frontend::onnx::DecoderBaseTensor> tensor_decoder;
};

// Stable sort by the model index projected from each element by key_of.
template <typename T, typename KeyFn>
void sort_by_model_index(std::vector<T>& items, KeyFn key_of) {
    std::stable_sort(items.begin(), items.end(), [&](const T& a, const T& b) {
        return key_of(a) < key_of(b);
    });
}
}  // namespace

void TranslateSession::add_result(const std::string& name,
                                  const ov::Output<ov::Node>& output_value,
                                  ResultVector& results) {
    const auto result = std::make_shared<ov::op::v0::Result>(output_value);
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

void TranslateSession::translate_op_and_store_outputs(const std::shared_ptr<DecoderBaseOperation>& op_decoder,
                                                      const std::shared_ptr<TelemetryExtension>& telemetry) {
    const auto out_size = op_decoder->get_output_size();
    ov::OutputVector ov_outputs = apply_op_translator(op_decoder, telemetry);
    for (size_t i = 0; i < ov_outputs.size() && i < out_size; ++i) {
        const auto& name = op_decoder->get_output_tensor_name(i);
        if (name.empty()) {  // not connected
            continue;
        }
        ov_outputs[i].add_names({name});
        m_tensor_values[name] = ov_outputs[i];
    }
}

void TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model,
                                       std::shared_ptr<ov::Model>& ov_model) {
    const auto model_onnx = std::dynamic_pointer_cast<unify::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(model_onnx != nullptr, "Invalid input model");

    const auto telemetry = model_onnx->get_telemetry_extension();
    ResultVector results;

    std::vector<std::pair<std::string, std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>>> graph_outputs;

    auto materialize_inputs_and_translate = [&](const std::shared_ptr<onnx::DecoderBaseOperation>& op_decoder,
                                                auto&& resolve_input_place) {
        FRONT_END_GENERAL_CHECK(op_decoder != nullptr, "Decoder must be onnx::DecoderBaseOperation or its child");
        for (size_t i = 0; i < op_decoder->get_input_size(); ++i) {
            const auto& name = op_decoder->get_input_tensor_name(i);
            // lookup_tensor() also checks m_tensor_values; materialize only on a full miss.
            if (name.empty() || lookup_tensor(name).get_node() != nullptr) {
                continue;
            }
            create_const_or_param(name, resolve_input_place(name, i));
        }
        translate_op_and_store_outputs(op_decoder, telemetry);
    };

    // Record a declared output as (name, data_place); keep the place only if it carries initializer data.
    auto collect_graph_output = [&](const std::string& name,
                                    const std::shared_ptr<ov::frontend::onnx::TensorONNXPlace>& data_place) {
        graph_outputs.emplace_back(name, data_place && place_has_data(*data_place) ? data_place : nullptr);
    };

    if (!model_onnx->is_loaded()) {
        // Not loaded to a Place-graph: walk the GraphIterator decoders directly (skips load_model()).
        const auto graph_iterator =
            std::dynamic_pointer_cast<ov::frontend::onnx::GraphIterator>(model_onnx->get_graph_iterator());
        FRONT_END_GENERAL_CHECK(graph_iterator != nullptr, "Invalid graph iterator for single-pass conversion");

        // Preserve zero-copy constant wrapping when the iterator's owner allows it.
        const bool reuse_const_data = model_onnx->is_const_data_reusable();
        static const std::string empty_tensor_name;

        auto make_transient_place = [&](const ov::frontend::onnx::TensorMetaInfo& info) {
            return make_transient_tensor_place(info, *model_onnx, reuse_const_data);
        };

        std::vector<DiscoveredOutput> discovered_outputs;
        // Model input index per graph-input Parameter, for reordering m_parameters after the walk.
        std::unordered_map<const ov::Node*, int64_t> input_indices;
        std::map<std::string, uint64_t> op_statistics;  // op_count telemetry, keyed to match load_model()

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
                    const auto node = create_const_or_param(name, make_transient_place(info));
                    // A no-data graph input is a Parameter unless it is the empty-constant special case.
                    if (ov::as_type_ptr<ov::op::v0::Parameter>(node)) {
                        input_indices[node.get()] = input_idx;
                    }
                }
                if (output_idx >= 0) {
                    discovered_outputs.push_back({output_idx, name, tensor_decoder});
                }
                continue;
            }

            const auto op_decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(decoder);
            materialize_inputs_and_translate(op_decoder, [&](const std::string&, size_t i) {
                return make_transient_place(op_decoder->get_input_tensor_info(i));
            });
            if (telemetry) {
                // Key by the iterator's resolved opset so op_count events match load_model()'s.
                op_statistics[op_decoder->get_op_type() + "-" +
                              std::to_string(graph_iterator->get_opset_version(op_decoder->get_domain()))]++;
            }
        }

        send_op_count_telemetry(telemetry, op_statistics);

        // Order graph-input Parameters by model input index. The Place-graph path gets this from
        // load_model()'s sorted get_inputs(); a custom GraphIterator may emit input decoders in any order.
        // Parent-scope Parameters (appended by lookup_tensor() during subgraph translation) are absent
        // from input_indices and sort to the back in their existing relative order.
        sort_by_model_index(m_parameters, [&](const std::shared_ptr<ov::op::v0::Parameter>& p) {
            const auto it = input_indices.find(p.get());
            return it != input_indices.end() ? it->second : std::numeric_limits<int64_t>::max();
        });

        sort_by_model_index(discovered_outputs, [](const DiscoveredOutput& o) {
            return o.model_output_index;
        });
        graph_outputs.reserve(discovered_outputs.size());
        for (const auto& discovered : discovered_outputs) {
            // Re-fetch from the still-alive decoder; the loop may have advanced (and freed) other decoders.
            const auto& info = discovered.tensor_decoder->get_tensor_info();
            collect_graph_output(discovered.tensor_name,
                                 tensor_has_data(info) ? make_transient_place(info) : nullptr);
        }
    } else {
        auto& all_tensor_places = model_onnx->get_tensor_places();

        // inputs
        m_parameters.reserve(model_onnx->get_inputs().size());
        for (const auto& input : model_onnx->get_inputs()) {
            const auto input_tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(input);
            FRONT_END_GENERAL_CHECK(input_tensor != nullptr,
                                    "Inputs of ov::frontend::onnx::InputModel must be TensorONNXPlace instances");
            create_const_or_param(input_tensor->get_names()[0], input_tensor);
        }

        // operations
        for (const auto& op_place : model_onnx->get_op_places()) {
            const auto op_decoder = std::dynamic_pointer_cast<onnx::DecoderBaseOperation>(op_place->get_decoder());
            materialize_inputs_and_translate(op_decoder, [&](const std::string& name, size_t) {
                const auto place_it = all_tensor_places.find(name);
                FRONT_END_GENERAL_CHECK(place_it != all_tensor_places.end(), "Tensor place not found in a graph");
                return place_it->second;
            });
        }

        graph_outputs.reserve(model_onnx->get_outputs().size());
        for (const auto& output : model_onnx->get_outputs()) {
            const auto tensor = std::dynamic_pointer_cast<ov::frontend::onnx::TensorONNXPlace>(output);
            FRONT_END_GENERAL_CHECK(tensor != nullptr,
                                    "Outputs of ov::frontend::onnx::InputModel must be TensorONNXPlace instances");
            const auto name = tensor->get_names()[0];
            const auto place_it = all_tensor_places.find(name);
            collect_graph_output(name, place_it != all_tensor_places.end() ? place_it->second : nullptr);
        }
    }

    results.reserve(graph_outputs.size());
    for (const auto& graph_output : graph_outputs) {
        const auto& name = graph_output.first;
        const auto& data_place = graph_output.second;
        if (!m_tensor_values.count(name)) {
            if (data_place) {
                create_const_or_param(name, data_place);
            } else {
                const auto parent_value = lookup_tensor(name);
                FRONT_END_GENERAL_CHECK(parent_value.get_node() != nullptr,
                                        "Output tensor \"",
                                        name,
                                        "\" was declared as a graph output but is not produced by any operation, "
                                        "is not an initializer, and cannot be resolved from a parent scope.");
                m_tensor_values.emplace(name, parent_value);
            }
        }
        add_result(name, m_tensor_values[name], results);
    }

    ov_model = std::make_shared<ov::Model>(results, m_parameters, "onnx_Frontend_IR");

    const auto metadata = model_onnx->get_graph_iterator()->get_metadata();
    const std::string framework_section = "framework";
    for (const auto& pair : metadata) {
        ov_model->set_rt_info(pair.second, framework_section, pair.first);
    }
}
