// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "helper_ops/gather_assign.hpp"
#include "helper_ops/slice_assign.hpp"
#include "input_model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch {

using namespace ov::op;

namespace {
// Alias views and their base values belong to the graph where they were converted.
class AliasScope {
public:
    explicit AliasScope(TranslateSession& session) : m_session(session) {
        m_aliases.swap(m_session.m_may_be_alias);
        m_pending_aliases.swap(m_session.m_pending_aliases);
    }

    ~AliasScope() {
        m_aliases.swap(m_session.m_may_be_alias);
        m_pending_aliases.swap(m_session.m_pending_aliases);
    }

private:
    TranslateSession& m_session;
    decltype(TranslateSession::m_may_be_alias) m_aliases;
    decltype(TranslateSession::m_pending_aliases) m_pending_aliases;
};

bool is_structured_value(const Output<Node>& value) {
    const auto node = value.get_node_shared_ptr();
    return ov::is_type<SequenceMark>(node) || ov::is_type<ComplexTypeMark>(node) ||
           ov::is_type<ov::op::util::FrameworkNode>(node);
}

Output<Node> get_numel(const Output<Node>& value) {
    const auto shape = std::make_shared<v3::ShapeOf>(value, element::i64);
    return std::make_shared<v1::ReduceProd>(shape, v0::Constant::create(element::i64, Shape{1}, {0}), false);
}

// Flat base indices of every element of `value`.
Output<Node> make_identity_positions(const Output<Node>& value) {
    const auto shape = std::make_shared<v3::ShapeOf>(value, element::i64);
    const auto range = std::make_shared<v4::Range>(v0::Constant::create(element::i64, Shape{}, {0}),
                                                   get_numel(value),
                                                   v0::Constant::create(element::i64, Shape{}, {1}),
                                                   element::i64);
    return std::make_shared<v1::Reshape>(range, shape, false);
}

// Maps -1 positions to the index of one padding element appended to the flattened base.
Output<Node> to_padded_index(const Output<Node>& positions, const Output<Node>& numel) {
    const auto is_missing = std::make_shared<v1::Less>(positions, v0::Constant::create(element::i64, Shape{}, {0}));
    return std::make_shared<v1::Select>(is_missing, numel, positions);
}

Output<Node> flatten_with_padding(const Output<Node>& value) {
    const auto flat = std::make_shared<v1::Reshape>(value, v0::Constant::create(element::i64, Shape{1}, {-1}), false);
    const auto pad = std::make_shared<v1::ConvertLike>(v0::Constant::create(element::i32, Shape{1}, {0}), value);
    return std::make_shared<v0::Concat>(OutputVector{flat, pad}, 0);
}

// Keeps an unsupported alias update in the graph, so conversion reports it instead of silently dropping it.
Output<Node> make_unsupported_alias_node(const std::shared_ptr<TorchDecoder>& decoder,
                                         const Output<Node>& value,
                                         const std::string& message) {
    const auto node = std::make_shared<PtFrameworkNode>(decoder, OutputVector{value}, 1, true, true);
    auto attrs = node->get_attrs();
    attrs[PtFrameworkNode::failed_conversion_key] = message;
    node->set_attrs(attrs);
    return node->output(0);
}

bool is_view_op(const std::shared_ptr<Node>& node) {
    return ov::is_type<v1::Reshape>(node) || ov::is_type<v0::Squeeze>(node) || ov::is_type<v0::Unsqueeze>(node) ||
           ov::is_type<v1::Transpose>(node) || ov::is_type<v8::Slice>(node) || ov::is_type<v1::StridedSlice>(node) ||
           ov::is_type<v8::Gather>(node) || ov::is_type<v1::Split>(node) || ov::is_type<v1::VariadicSplit>(node) ||
           ov::is_type<v3::Broadcast>(node) || ov::is_type<v1::Broadcast>(node);
}

bool is_identity_convert(const std::shared_ptr<Node>& node) {
    if (!ov::is_type<v0::Convert>(node) && !ov::is_type<v1::ConvertLike>(node)) {
        return false;
    }
    const auto& in_type = node->get_input_element_type(0);
    const auto& out_type = node->get_output_element_type(0);
    return in_type.is_dynamic() || out_type.is_dynamic() || in_type == out_type;
}

// Replays the operations between a base and its view on positions of the base. Only data movement operations are
// allowed on the data path; values derived from the base shape are replayed unchanged.
struct PositionsReplay {
    std::map<Output<Node>, Output<Node>> data;
    std::map<Output<Node>, Output<Node>> shape_derived;

    bool replay(const Output<Node>& value, Output<Node>& replayed, bool& is_data) {
        if (const auto found = data.find(value); found != data.end()) {
            replayed = found->second;
            is_data = true;
            return true;
        }
        if (const auto found = shape_derived.find(value); found != shape_derived.end()) {
            replayed = found->second;
            is_data = false;
            return true;
        }
        const auto node = value.get_node_shared_ptr();
        if (ov::is_type<v0::Parameter>(node) || ov::is_type<v0::Constant>(node)) {
            replayed = value;
            is_data = false;
            return true;
        }
        OutputVector new_inputs;
        std::vector<size_t> data_ports;
        bool changed = false;
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            Output<Node> input;
            bool input_is_data = false;
            if (!replay(node->input_value(i), input, input_is_data)) {
                return false;
            }
            changed = changed || input != node->input_value(i);
            if (input_is_data) {
                data_ports.push_back(i);
            }
            new_inputs.push_back(input);
        }
        is_data = !data_ports.empty();
        if (is_data) {
            if (data_ports != std::vector<size_t>{0}) {
                return false;
            }
            if (ov::is_type<v3::ShapeOf>(node) || ov::is_type<v0::ShapeOf>(node)) {
                is_data = false;
            } else if (is_identity_convert(node)) {
                data[value] = new_inputs[0];
                replayed = new_inputs[0];
                return true;
            } else if (!is_view_op(node)) {
                return false;
            }
        }
        if (!changed) {
            replayed = value;
            (is_data ? data : shape_derived)[value] = value;
            return true;
        }
        const auto updated = node->clone_with_new_inputs(new_inputs);
        for (size_t i = 0; i < node->get_output_size(); ++i) {
            (is_data ? data : shape_derived)[node->output(i)] = updated->output(i);
        }
        replayed = updated->output(value.get_index());
        return true;
    }
};

// Returns positions of `view` in `base`, or an empty output if the view cannot be expressed as positions.
Output<Node> replay_positions(const Output<Node>& view, const Output<Node>& base, const Output<Node>& base_positions) {
    if (is_structured_value(view) || is_structured_value(base)) {
        return {};
    }
    PositionsReplay replay;
    replay.data[base] = base_positions;
    Output<Node> positions;
    bool is_data = false;
    if (!replay.replay(view, positions, is_data) || !is_data) {
        return {};
    }
    return positions;
}

// Helper to extract complex part element type from raw type
element::Type get_complex_part_type(const Any& raw_type) {
    element::Type complex_part_type = element::dynamic;
    if (raw_type.is<type::Complex>()) {
        const auto& complex_type = raw_type.as<type::Complex>();
        if (complex_type.element_type.is<element::Type>()) {
            complex_part_type = complex_type.element_type.as<element::Type>();
        }
    } else if (raw_type.is<type::Tensor>()) {
        const auto& tensor_type = raw_type.as<type::Tensor>();
        if (tensor_type.element_type.is<type::Complex>()) {
            const auto& complex_type = tensor_type.element_type.as<type::Complex>();
            if (complex_type.element_type.is<element::Type>()) {
                complex_part_type = complex_type.element_type.as<element::Type>();
            }
        }
    }
    return complex_part_type;
}

// Helper to create parameter and register it in tensor_map, wrapping in ComplexTypeMark if needed
void register_parameter_in_tensor_map(const std::shared_ptr<v0::Parameter>& parameter,
                                      size_t input_id,
                                      const Any& type_any,
                                      const Any& raw_type,
                                      const std::shared_ptr<std::unordered_map<size_t, Output<Node>>>& tensor_map) {
    if (type_any.is<type::Complex>()) {
        auto complex_part_type = get_complex_part_type(raw_type);
        auto complex_mark = std::make_shared<ComplexTypeMark>(parameter->output(0), complex_part_type);
        (*tensor_map)[input_id] = complex_mark;
    } else {
        (*tensor_map)[input_id] = parameter;
    }
}
}  // namespace

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::unordered_map<std::string, CreatorFunction>& translator_map,
                                   const std::shared_ptr<TelemetryExtension>& telemetry)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_telemetry(telemetry),
      m_ov_model(nullptr) {}

TranslateSession::~TranslateSession() {
    if (m_telemetry) {
        // Send statistics
        for (const auto& op : m_op_statistics) {
            m_telemetry->send_event("op_count", "pytorch_" + op.first, static_cast<int>(op.second));
        }
    }
}

std::shared_ptr<ov::Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<ov::Model> TranslateSession::translate_graph(const ov::frontend::InputModel::Ptr& input_model) {
    auto pytorch_model = std::dynamic_pointer_cast<pytorch::InputModel>(input_model);
    FRONT_END_GENERAL_CHECK(pytorch_model != nullptr, "Invalid input model");
    auto model = convert_pytorch_model(pytorch_model->m_model_decoder, {}, pytorch_model);
    // First delete tensor indexes from outputs then resolve input names, otherwise Parameter->Result will fail
    std::set<Output<Node>> named_outputs;
    for (auto& result : model->get_results()) {
        auto tensor_desc = result->input_value(0);
        if (!named_outputs.insert(tensor_desc).second) {
            continue;
        }
        auto names = tensor_desc.get_names();
        if (!names.empty()) {
            auto tensor_idx = decode_tensor_name(tensor_desc);
            if (names.erase(std::to_string(tensor_idx))) {
                tensor_desc.set_names(names);
            }
        }
    }
    // Set input tensor names to be equal to signature name saved in friendly name
    for (auto& param : model->get_parameters()) {
        if (param->get_friendly_name() != param->get_name()) {
            // get_name is autogenerated name, we need to make sure that this parameter was named by frontend
            param->output(0).set_names({param->get_friendly_name()});
        }
    }

    // process model rt_info
    auto rt_info = pytorch_model->get_decoder()->get_rt_info();
    for (const auto& item : rt_info) {
        model->set_rt_info(item.second, item.first);
    }
    model->set_rt_info(ov::Any(pytorch_model->decoder_type_name()), "decoder_type_name");

    return model;
}

std::shared_ptr<Model> TranslateSession::convert_pytorch_model(
    std::shared_ptr<TorchDecoder> pytorch_model,
    const TensorMap& external_tensor_map,
    const std::shared_ptr<pytorch::InputModel>& input_model) {
    AliasScope alias_scope(*this);
    // The decoder type is a property of the decoder class, so read it once per graph rather
    // than per node; every access crosses into Python.
    m_is_fx = pytorch_model->decoder_type_name() == "fx";
    std::shared_ptr<Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        auto parameters = std::make_shared<ParameterVector>();
        auto tensor_map = std::make_shared<TensorMap>();  // tensor map of the current context
        auto mutated_tensors = std::make_shared<std::set<size_t>>();
        std::vector<size_t> inserted_params;

        if (input_model && input_model->m_requested_places.size() == 0) {
            // When we have input model we should use its inputs order to create Parameters
            // We use m_inputs instead of get_inputs() because latter doesn't have "self" input
            // If there are fake places we don't need to use model inputs in that case
            for (auto& input_p : input_model->m_inputs) {
                auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(input_p);
                FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported.");
                auto tensor_id = pytorch_place->get_tensor_index();
                element::Type type = pytorch_place->get_element_type();
                PartialShape pshape = pytorch_place->get_partial_shape();
                auto parameter = std::make_shared<v0::Parameter>(type, pshape);
                if (!pytorch_place->get_names().empty())
                    parameter->set_friendly_name(pytorch_place->get_names().front());
                encode_tensor_name(parameter->output(0), tensor_id);
                parameters->emplace_back(parameter);
                (*tensor_map)[tensor_id] = parameter;
            }
        } else {
            // Go over all pytorch_model inputs and register them in the tensor map:
            auto inputs = pytorch_model->inputs();
            for (size_t i = 0; i < inputs.size(); ++i) {
                element::Type type = element::dynamic;
                PartialShape pshape = pytorch_model->get_input_shape(i);
                auto raw_type = pytorch_model->get_input_type(i);
                auto type_any = simplified_type_interpret(raw_type);
                // TODO: Use special API to set custom type specification
                if (type_any.is<element::Type>()) {
                    type = type_any.as<element::Type>();
                }
                auto parameter = std::make_shared<v0::Parameter>(type, pshape);
                parameter->set_friendly_name(pytorch_model->get_input_signature_name(i));
                encode_tensor_name(parameter->output(0), inputs.at(i), {pytorch_model->get_input_debug_name(i)});
                parameters->emplace_back(parameter);

                register_parameter_in_tensor_map(parameter, inputs.at(i), type_any, raw_type, tensor_map);
            }
        }
        if (input_model) {
            // Add all tensors that were frozen
            for (auto& desc : input_model->m_descriptors) {
                (*tensor_map)[desc.first] = desc.second.m_value;
            }
        }

        auto node_visitor = [&](std::shared_ptr<TorchDecoder> node) {
            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we create new input for the model to
            // link with external scope on a higher level.

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = raw_inputs.at(i);
                // If inputs are inlined (possible only for fx decoder) we shouldn't add a Parameter for it
                if (input == 0 && node->is_input_inlined(i)) {
                    continue;
                }
                if (tensor_map->find(input) == tensor_map->end()) {
                    // Input refers value in the outer scope, need to create a new Parameter in the current scope
                    // Linkage to external scope will be performed on the level of the parent operation (if or loop)
                    // TODO: Eliminate duplication with the main code for Parameters creation
                    PartialShape ps = node->get_input_shape(i);
                    auto raw_type = node->get_input_type(i);
                    auto type = simplified_type_interpret(raw_type);
                    auto dtype = element::dynamic;
                    if (type.is<element::Type>()) {
                        dtype = type.as<element::Type>();
                    }
                    auto parameter = std::make_shared<v0::Parameter>(dtype, ps);
                    // set name of parameter to the index of node in the model
                    encode_tensor_name(parameter->output(0), input);
                    parameters->push_back(parameter);
                    inserted_params.push_back(input);

                    register_parameter_in_tensor_map(parameter, input, type, raw_type, tensor_map);
                }
            }
            auto context = NodeContext(node, external_tensor_map, tensor_map, parameters, mutated_tensors, this);
            // Add op type in the statistics
            m_op_statistics[context.get_op_type()]++;
            auto converted_outputs = convert_node(context);
            const bool has_out = m_is_fx && context.has_attribute("out");
            if (has_out) {
                FRONT_END_OP_CONVERSION_CHECK(converted_outputs.size() == 1, "Expected a single out tensor.");
                const auto out = context.get_input("out");
                converted_outputs[0] = ComplexTypeMark::convert_like(context, converted_outputs[0], out);
                context.mutate_input("out", converted_outputs[0]);
            }

            const auto fw_outputs = context.outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after conversion compared to pytorch ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          context.get_op_type(),
                                          " outputs greater than number of converted outputs, which are",
                                          fw_outputs.size(),
                                          " and ",
                                          converted_outputs.size(),
                                          " respectively.");

            const bool has_inputs = !raw_inputs.empty();
            const size_t first_input_id = has_out ? node->get_named_input("out") : (has_inputs ? raw_inputs.at(0) : 0);
            const auto& op_type = context.get_op_type();
            const bool constructs_sequence = op_type == "prim::TupleConstruct" || op_type == "prim::ListConstruct";
            std::vector<size_t> unpacked_ids;
            const auto sequence = m_may_be_alias.find(first_input_id);
            if (sequence != m_may_be_alias.end() && !sequence->second.element_ids.empty()) {
                const auto& elements = sequence->second.element_ids;
                if (op_type == "prim::TupleUnpack" || op_type == "prim::ListUnpack") {
                    unpacked_ids = elements;
                } else if (op_type == "aten::__getitem__") {
                    if (const auto index_node = ov::util::get_constant_from_source(context.get_input(1))) {
                        auto index = index_node->cast_vector<int64_t>().at(0);
                        if (index < 0) {
                            index += static_cast<int64_t>(elements.size());
                        }
                        if (index >= 0 && static_cast<size_t>(index) < elements.size()) {
                            unpacked_ids = {elements[index]};
                        }
                    }
                }
            }
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = fw_outputs.at(i);
                const auto in_tensor_id = unpacked_ids.empty() ? first_input_id : unpacked_ids.at(i);
                if (has_out || (has_inputs && (constructs_sequence || node->may_produce_alias(0, i)))) {
                    auto alias_iter = m_may_be_alias.find(fw_tensor_id);
                    // TODO: do we need to check other inputs, not only 0?
                    if (alias_iter != m_may_be_alias.end()) {
                        const auto& recorded = alias_iter->second;
                        FRONT_END_GENERAL_CHECK(recorded.base_id == in_tensor_id,
                                                "Operation ",
                                                context.get_op_type(),
                                                " creates alias to tensor which was already created before by ",
                                                recorded.decoder->get_op_type(),
                                                ", but from different tensor: ",
                                                in_tensor_id,
                                                " vs ",
                                                recorded.base_id);
                    }
                    m_may_be_alias[fw_tensor_id] = {in_tensor_id,
                                                    node,
                                                    converted_outputs[i],
                                                    tensor_map->at(in_tensor_id),
                                                    constructs_sequence ? raw_inputs : std::vector<size_t>{}};
                    OPENVINO_DEBUG("Registered alias: ",
                                   fw_tensor_id,
                                   " of tensor: ",
                                   in_tensor_id,
                                   " of operation: ",
                                   context.get_op_type());
                }
                FRONT_END_GENERAL_CHECK(tensor_map->find(fw_tensor_id) == tensor_map->end(),
                                        "Duplicated producer for PT value with unique ID: ",
                                        fw_tensor_id);

#ifdef ENABLE_OPENVINO_DEBUG
                const auto out_type = simplified_type_interpret(context.get_output_type(i));
                if (out_type.is<element::Type>()) {
                    if (!converted_outputs[i].get_element_type().compatible(out_type.as<element::Type>())) {
                        OPENVINO_DEBUG("[WARNING] Produced output type for operation ",
                                       context.get_op_type(),
                                       " for tensor id: ",
                                       fw_tensor_id,
                                       " is incompatible: produced ",
                                       converted_outputs[i].get_element_type(),
                                       " vs ",
                                       out_type.as<element::Type>());
                    }
                }
#endif
                if (!m_may_be_alias.count(fw_tensor_id)) {
                    const auto pending = m_pending_aliases.find(converted_outputs[i]);
                    // Only outputs of the subgraph operation and elements selected from them are the view. Other
                    // operations, like clone, may return the same OpenVINO value for a new tensor.
                    const auto is_subgraph_output = [&](const PendingAlias& alias) {
                        if (alias.decoder == node) {
                            return true;
                        }
                        if (!has_inputs ||
                            (op_type != "aten::__getitem__" && op_type != "<built-in function getitem>" &&
                             op_type != "prim::TupleUnpack" && op_type != "prim::ListUnpack")) {
                            return false;
                        }
                        const auto outputs = alias.decoder->outputs();
                        return std::find(outputs.begin(), outputs.end(), raw_inputs.at(0)) != outputs.end();
                    };
                    if (pending != m_pending_aliases.end() && is_subgraph_output(pending->second)) {
                        const auto& alias = pending->second;
                        m_may_be_alias[fw_tensor_id] =
                            {alias.base_id, alias.decoder, converted_outputs[i], alias.base_value, {}, alias.positions};
                        OPENVINO_DEBUG("Registered subgraph output alias: ",
                                       fw_tensor_id,
                                       " of tensor: ",
                                       alias.base_id);
                    }
                }
                (*tensor_map)[fw_tensor_id] = converted_outputs[i];
                encode_tensor_name(converted_outputs[i], fw_tensor_id, {node->get_output_debug_name(i)});
            }
        };

        FRONT_END_GENERAL_CHECK(pytorch_model->decoder_type_name() != "ts" || pytorch_model->get_subgraph_size() == 1,
                                "Model should have exactly 1 subgraph for TorchScript.");
        pytorch_model->visit_subgraph(node_visitor);

        NodeContext output_context(pytorch_model, external_tensor_map, tensor_map, parameters, mutated_tensors, this);
        ResultVector results;
        if (input_model) {
            // For the case when we have InputModel we need to have same order as its outputs
            for (auto& output_p : input_model->get_outputs()) {
                auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(output_p);
                FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported.");
                auto tensor_id = pytorch_place->get_tensor_index();
                auto ov_output = output_context.resolve_tensor(tensor_id);
                FRONT_END_GENERAL_CHECK(!ov_output.get_names().empty(),
                                        "Tensor doesn't have name, while it should have name: ",
                                        tensor_id);
                auto result = std::make_shared<v0::Result>(ov_output);
                results.push_back(result);
            }
        } else {
            for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
                size_t id = pytorch_model->output(i);
                auto it = tensor_map->find(id);
                if (it == tensor_map->end()) {
                    // Not found in this scope, adding Parameter to connect to external scope
                    auto parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                    encode_tensor_name(parameter->output(0), id);
                    parameters->push_back(parameter);
                    it = tensor_map->emplace(id, parameter).first;
                }
                FRONT_END_GENERAL_CHECK(!it->second.get_names().empty(),
                                        "Tensor doesn't have name, while it should have name: ",
                                        id);
                auto result = std::make_shared<v0::Result>(output_context.resolve_tensor(id));
                results.push_back(result);
            }
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (const auto& param : *parameters) {
            auto input_idx = decode_tensor_name(param->output(0));
            param_names.insert(input_idx);
        }
        for (const auto& tensor_id : *mutated_tensors) {
            if (param_names.count(tensor_id)) {
                FRONT_END_GENERAL_CHECK(tensor_map->count(tensor_id),
                                        "Tensor with id: ",
                                        tensor_id,
                                        " doesn't exist in tensor map.");
                // model input was mutated we need to make a result for it
                // empty external_tensor_map means this is main body of the model and we don't want to create
                // additional outputs in that case.
                if (!external_tensor_map.empty()) {
                    OPENVINO_DEBUG("Creating Result for mutated tensor  ", tensor_id);
                    results.push_back(std::make_shared<v0::Result>(tensor_map->at(tensor_id)));
                }
            } else {
                OPENVINO_DEBUG("Mutated tensor with id ", tensor_id, " doesn't exist in inputs, skipping.");
            }
        }
        std::vector<SubgraphOutputAlias> output_aliases;
        if (!external_tensor_map.empty() && !input_model) {
            // Record declared outputs which are views of body inputs, so the parent can keep the alias relation.
            for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
                const auto id = pytorch_model->output(i);
                const auto root = get_alias_root(id);
                const auto value = results[i]->input_value(0);
                if (!param_names.count(root) || is_structured_value(value)) {
                    continue;
                }
                std::vector<AliasInfo> chain;
                const auto relation = get_alias_chain(id, root, value, chain);
                auto positions = std::make_shared<AliasPositions>([relation, chain, value]() -> Output<Node> {
                    return relation == AliasRelation::ALIAS ? compute_alias_positions(chain, value) : Output<Node>{};
                });
                output_aliases.push_back({i, root, positions});
            }
        }
        if (!external_tensor_map.empty()) {
            // for internal bodies we want to remove all extra inputs that were created, but not used
            parameters->erase(std::remove_if(parameters->begin(),
                                             parameters->end(),
                                             [&](std::shared_ptr<v0::Parameter> p) {
                                                 auto tensor_id = decode_tensor_name(p);
                                                 return p->output(0).get_target_inputs().empty() &&
                                                        std::find(inserted_params.begin(),
                                                                  inserted_params.end(),
                                                                  tensor_id) != inserted_params.end();
                                             }),
                              parameters->end());
        }
        resulting_model = std::make_shared<Model>(results, *parameters);
        if (!output_aliases.empty()) {
            m_subgraph_output_aliases[resulting_model.get()] = {resulting_model, std::move(output_aliases)};
        }
        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    return resulting_model;
}

OutputVector TranslateSession::convert_node(const NodeContext& context) {
    std::string exception;
    try {
        const auto& raw_op_type = context.get_op_type();
        auto it = m_translator_map.find(raw_op_type);
        // FX names carry an overload suffix; fall back to the shared TorchScript translator for the same operator.
        // Stays empty unless that fallback is needed, so the common path does not copy the name.
        std::string canonical_op_type;
        if (it == m_translator_map.end()) {
            canonical_op_type = normalize_op_type(raw_op_type);
            it = m_translator_map.find(canonical_op_type);
        }
        const std::string& op_type = canonical_op_type.empty() ? raw_op_type : canonical_op_type;
        if (it != m_translator_map.end()) {
            auto outputs = it->second(context);
            // FX represents multiple operator results as a single tuple value.
            if (outputs.size() > 1 && m_is_fx) {
                return {context.mark_node(make_list_construct(outputs))};
            }
            return outputs;
        } else if (op_type.back() == '_') {
            // inplace op case
            std::string op_type_cut = op_type.substr(0, op_type.size() - 1);
            auto it = m_translator_map.find(op_type_cut);
            if (it != m_translator_map.end()) {
                const auto& res = it->second(context);
                FRONT_END_OP_CONVERSION_CHECK(res.size() == 1, "inplace op must have single output.");
                context.mutate_input(0, res[0]);
                return res;
            }
        }
        OPENVINO_DEBUG("No translator found for: ", op_type, "\n");
    } catch (std::exception& e) {
        exception = e.what();
    } catch (...) {
        exception = "Unknown exception type.";
    }
    OPENVINO_DEBUG(exception, "\n");
    try {
        // Create PtFrameworkNode for everything that wasn't able to be converted normally
        return make_framework_node(context, exception);
    } catch (std::exception& e) {
        exception += " Exception happened while creating FrameworkNode with subgraphs: " + std::string(e.what());
    } catch (...) {
        exception += " Unknown exception happened while creating FrameworkNode with subgraphs";
    }
    OPENVINO_DEBUG(exception, "\n");
    return make_framework_node_ignore_bodies(context, exception);
}

void TranslateSession::encode_tensor_name(Output<Node> output,
                                          size_t tensor_idx,
                                          const std::vector<std::string>& additional_names) {
    if (!output.get_names().empty()) {
        OPENVINO_DEBUG("Tensor names already exist: ",
                       output.get_any_name(),
                       ". Will not be rewritten with ",
                       tensor_idx,
                       ". This is likely a mutated tensor.");
        return;
    }
    auto name = std::to_string(tensor_idx);
    std::unordered_set<std::string> names = {name};
    if (!additional_names.empty()) {
        names.insert(additional_names.begin(), additional_names.end());
    }

    auto it = m_counter_map.find(tensor_idx);
    if (it != m_counter_map.end()) {
        auto& pair = it->second;
        auto new_name = name + '_' + std::to_string(++pair.first);
        pair.second.set_names({std::move(new_name)});
        pair.second = output;
    } else {
        m_counter_map.emplace(tensor_idx, std::make_pair(0, output));
    }
    output.set_names(std::move(names));
}

namespace {
bool is_number(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}
}  // namespace

size_t TranslateSession::decode_tensor_name(const Output<Node>& output) {
    // any_name should always return numerical value even if there is a word value exist in names
    auto name = output.get_any_name();
    auto pos = name.find("_");
    if (pos != std::string::npos) {
        name = name.substr(0, pos);
    }
    // numbers after "_" will be ignored by stoll function
    FRONT_END_GENERAL_CHECK(is_number(name), "Tensor name is not a number: ", name);
    return static_cast<size_t>(std::stoll(name));
}

namespace {
Output<Node> slice_reverseprop(const Output<Node>& slice_output, const Output<Node>& value) {
    auto slice_node = slice_output.get_node_shared_ptr();
    FRONT_END_OP_CONVERSION_CHECK(ov::as_type_ptr<v8::Slice>(slice_node),
                                  "Conversion rule for aten::slice doesn't contain Slice node.");

    auto to_insert_data = slice_node->input_value(0);
    Output<Node> res;
    if (slice_node->get_input_size() == 5) {
        res = std::make_shared<SliceAssign>(to_insert_data,
                                            value,
                                            slice_node->input_value(1),
                                            slice_node->input_value(2),
                                            slice_node->input_value(3),
                                            slice_node->input_value(4));
    } else if (slice_node->get_input_size() == 4) {
        res = std::make_shared<SliceAssign>(to_insert_data,
                                            value,
                                            slice_node->input_value(1),
                                            slice_node->input_value(2),
                                            slice_node->input_value(3));
    } else {
        FRONT_END_OP_CONVERSION_CHECK(false, "Incorrect number of Slice inputs");
    }

    return res;
}

Output<Node> select_reverseprop(const Output<Node>& select_output, const Output<Node>& value) {
    auto gather_node = select_output.get_node_shared_ptr();
    FRONT_END_OP_CONVERSION_CHECK(ov::as_type_ptr<v8::Gather>(gather_node),
                                  "Conversion rule for aten::select doesn't contain Gather node.");

    auto to_insert_data = gather_node->input_value(0);
    return std::make_shared<GatherAssign>(to_insert_data,
                                          value,
                                          gather_node->input_value(1),
                                          gather_node->input_value(2));
}
}  // namespace

Output<Node> TranslateSession::get_reverseprop_op(const std::shared_ptr<TorchDecoder>& node,
                                                  const Output<Node>& direct_op_output,
                                                  const Output<Node>& value,
                                                  const Output<Node>& base) {
    try {
        const auto direct_node = direct_op_output.get_node_shared_ptr();
        if (base.get_node()) {
            if (direct_op_output == base) {
                return value;
            }
            if (const auto sequence = ov::as_type_ptr<SequenceMark>(base.get_node_shared_ptr())) {
                auto elements = sequence->get_sequence();
                for (auto& element : elements) {
                    if (direct_op_output == element) {
                        element = value;
                        return make_list_construct(elements);
                    }
                }
            }
            if (const auto sequence = ov::as_type_ptr<SequenceMark>(direct_node)) {
                const auto updated_sequence = ov::as_type_ptr<SequenceMark>(value.get_node_shared_ptr());
                FRONT_END_OP_CONVERSION_CHECK(updated_sequence, "Expected an aliased list update.");
                const auto elements = sequence->get_sequence();
                const auto updates = updated_sequence->get_sequence();
                FRONT_END_OP_CONVERSION_CHECK(elements.size() == updates.size(),
                                              "An aliased list cannot change length.");
                Output<Node> updated_base;
                for (size_t i = 0; i < elements.size(); ++i) {
                    if (elements[i] != updates[i]) {
                        FRONT_END_OP_CONVERSION_CHECK(!updated_base.get_node(), "Expected one updated list element.");
                        updated_base = get_reverseprop_op(node, elements[i], updates[i], base);
                    }
                }
                return updated_base.get_node() ? updated_base : base;
            }
            if (const auto complex_base = ov::as_type_ptr<ComplexTypeMark>(base.get_node_shared_ptr())) {
                if (direct_op_output == complex_base->get_data()) {
                    return std::make_shared<ComplexTypeMark>(value, value.get_element_type());
                }
                if (direct_op_output == complex_base->get_real()) {
                    return std::make_shared<ComplexTypeMark>(value, complex_base->get_imag());
                }
                if (direct_op_output == complex_base->get_imag()) {
                    return std::make_shared<ComplexTypeMark>(complex_base->get_real(), value);
                }
            }
            if (const auto complex_view = ov::as_type_ptr<ComplexTypeMark>(direct_node)) {
                const auto complex_value = ov::as_type_ptr<ComplexTypeMark>(value.get_node_shared_ptr());
                FRONT_END_OP_CONVERSION_CHECK(complex_value, "Expected a complex alias update.");
                return get_reverseprop_op(node, complex_view->get_data(), complex_value->get_data(), base);
            }
            const auto updated = get_reverseprop_op(node, direct_op_output, value);
            if (ov::as_type_ptr<PtFrameworkNode>(updated.get_node_shared_ptr())) {
                return updated;
            }
            return get_reverseprop_op(node, direct_node->input_value(0), updated, base);
        }
        if (ov::is_type<v0::Convert>(direct_node) || ov::is_type<v1::ConvertLike>(direct_node)) {
            return value;
        }
        if (ov::is_type<v8::Slice>(direct_node)) {
            return slice_reverseprop(direct_op_output, value);
        }
        if (ov::is_type<v8::Gather>(direct_node)) {
            return select_reverseprop(direct_op_output, value);
        }
        if (ov::is_type<v1::Split>(direct_node) || ov::is_type<v1::VariadicSplit>(direct_node)) {
            const auto axis = ov::util::get_constant_from_source(direct_node->input_value(1));
            FRONT_END_OP_CONVERSION_CHECK(axis, "Cannot reverse a split with a dynamic axis.");
            auto outputs = direct_node->outputs();
            outputs[direct_op_output.get_index()] = value;
            return std::make_shared<v0::Concat>(outputs, axis->cast_vector<int64_t>().at(0));
        }
        if (ov::is_type<v1::Reshape>(direct_node) || ov::is_type<v0::Squeeze>(direct_node) ||
            ov::is_type<v0::Unsqueeze>(direct_node)) {
            return std::make_shared<v1::Reshape>(value,
                                                 std::make_shared<v3::ShapeOf>(direct_node->input_value(0)),
                                                 false);
        }
        if (ov::is_type<v1::Transpose>(direct_node)) {
            const auto order = direct_node->input_value(1);
            const auto zero = v0::Constant::create(element::i32, Shape{}, {0});
            const auto one = v0::Constant::create(element::i32, Shape{}, {1});
            const auto shape = std::make_shared<v3::ShapeOf>(order, element::i32);
            const auto rank = std::make_shared<v8::Gather>(shape, zero, zero);
            const auto axes = std::make_shared<v4::Range>(zero, rank, one, order.get_element_type());
            // inverse[order[i]] = i, including permutations built from a dynamic rank.
            const auto inverse = std::make_shared<v3::ScatterElementsUpdate>(axes, order, axes, zero);
            return std::make_shared<v1::Transpose>(value, inverse);
        }

    }
#ifdef ENABLE_OPENVINO_DEBUG
    catch (std::exception& e) {
        OPENVINO_DEBUG("Exception happened during conversion of backprop op: ",
                       node->get_op_type(),
                       " with schema: ",
                       node->get_schema(),
                       ": ",
                       e.what());
    }
#else
    catch (std::exception&) {
    }
#endif
    // Create PtFrameworkNode representing unconverted backprop operation
    return std::make_shared<PtFrameworkNode>(node, OutputVector{value}, 1, true);
}

Output<Node> make_non_alias_positions(const Output<Node>& value) {
    return std::make_shared<v3::Broadcast>(v0::Constant::create(element::i64, Shape{}, {-1}),
                                           std::make_shared<v3::ShapeOf>(value, element::i64));
}

Output<Node> make_alias_identity_positions(const Output<Node>& value) {
    return make_identity_positions(value);
}

Output<Node> replay_alias_positions(const Output<Node>& view,
                                    const Output<Node>& base,
                                    const Output<Node>& base_positions) {
    return replay_positions(view, base, base_positions);
}

Output<Node> compose_alias_positions(const Output<Node>& outer, const Output<Node>& inner) {
    return std::make_shared<v8::Gather>(flatten_with_padding(outer),
                                        to_padded_index(inner, get_numel(outer)),
                                        v0::Constant::create(element::i64, Shape{}, {0}));
}

Output<Node> TranslateSession::AliasPositions::get() {
    if (!m_materialized) {
        m_positions = m_materializer();
        m_materialized = true;
        m_materializer = nullptr;
    }
    return m_positions;
}

std::vector<TranslateSession::SubgraphOutputAlias> TranslateSession::take_subgraph_output_aliases(
    const std::shared_ptr<Model>& body) {
    const auto found = m_subgraph_output_aliases.find(body.get());
    if (found == m_subgraph_output_aliases.end()) {
        return {};
    }
    std::vector<SubgraphOutputAlias> aliases;
    if (found->second.first.lock() == body) {
        aliases = std::move(found->second.second);
    }
    m_subgraph_output_aliases.erase(found);
    return aliases;
}

void TranslateSession::register_output_alias(const Output<Node>& output,
                                             size_t base_id,
                                             const Output<Node>& base_value,
                                             const std::shared_ptr<TorchDecoder>& decoder,
                                             const std::shared_ptr<AliasPositions>& positions) {
    m_pending_aliases[output] = {base_id, base_value, decoder, positions};
}

size_t TranslateSession::get_alias_root(size_t tensor_id) const {
    std::set<size_t> visited;
    auto alias = m_may_be_alias.find(tensor_id);
    while (alias != m_may_be_alias.end() && alias->second.element_ids.empty() && visited.insert(tensor_id).second) {
        tensor_id = alias->second.base_id;
        alias = m_may_be_alias.find(tensor_id);
    }
    return tensor_id;
}

TranslateSession::AliasRelation TranslateSession::get_alias_chain(size_t tensor_id,
                                                                  size_t root_id,
                                                                  const Output<Node>& value,
                                                                  std::vector<AliasInfo>& chain) const {
    chain.clear();
    const auto root = get_alias_root(tensor_id);
    if (root != root_id) {
        return tensor_id == root && !is_structured_value(value) ? AliasRelation::NONE : AliasRelation::UNSUPPORTED;
    }
    for (auto id = tensor_id; id != root_id; id = m_may_be_alias.at(id).base_id) {
        chain.push_back(m_may_be_alias.at(id));
    }
    const auto& root_value = chain.empty() ? value : chain.back().base_value;
    return is_structured_value(root_value) ? AliasRelation::UNSUPPORTED : AliasRelation::ALIAS;
}

Output<Node> TranslateSession::compute_alias_positions(const std::vector<AliasInfo>& chain, const Output<Node>& value) {
    if (chain.empty()) {
        return make_identity_positions(value);
    }
    // An empty output stands for identity positions of the root until a link needs them.
    Output<Node> positions;
    for (auto link = chain.rbegin(); link != chain.rend(); ++link) {
        if (link->positions) {
            // Positions of the link are relative to its base, which is the previous tensor in the chain.
            const auto relative = link->positions->get();
            if (!relative.get_node()) {
                return {};
            }
            positions = positions.get_node() ? compose_alias_positions(positions, relative) : relative;
        } else {
            const auto base_positions = positions.get_node() ? positions : make_identity_positions(link->base_value);
            positions = replay_positions(link->output, link->base_value, base_positions);
            if (!positions.get_node()) {
                return {};
            }
        }
    }
    return positions;
}

Output<Node> TranslateSession::reverseprop_alias(const AliasInfo& alias_info, const Output<Node>& value) {
    if (!alias_info.positions) {
        return get_reverseprop_op(alias_info.decoder, alias_info.output, value, alias_info.base_value);
    }
    const auto positions = alias_info.positions->get();
    if (!positions.get_node() || is_structured_value(alias_info.base_value) || is_structured_value(value)) {
        return make_unsupported_alias_node(alias_info.decoder,
                                           value,
                                           "Cannot propagate a mutation of a view returned from " +
                                               alias_info.decoder->get_op_type() + " to its base tensor.");
    }
    // Index of the written element for each base element, -1 where the base is not written. Only indices are
    // padded, so base and value data are never copied into a larger buffer.
    const auto& base = alias_info.base_value;
    const auto shape = std::make_shared<v3::ShapeOf>(base, element::i64);
    const auto numel = get_numel(base);
    const auto flat_shape = v0::Constant::create(element::i64, Shape{1}, {-1});
    const auto minus_one = v0::Constant::create(element::i64, Shape{}, {-1});
    const auto zero = v0::Constant::create(element::i64, Shape{}, {0});
    const auto one = v0::Constant::create(element::i64, Shape{}, {1});
    const auto indices = std::make_shared<v0::Unsqueeze>(
        std::make_shared<v1::Reshape>(to_padded_index(positions, numel), flat_shape, false),
        minus_one);
    const auto source = std::make_shared<v4::Range>(zero, get_numel(positions), one, element::i64);
    const auto padded_size = std::make_shared<v1::Add>(numel, one);
    const auto no_source =
        std::make_shared<v3::Broadcast>(minus_one, std::make_shared<v0::Unsqueeze>(padded_size, zero));
    const auto padded_sources = std::make_shared<v3::ScatterNDUpdate>(no_source, indices, source);
    const auto sources = std::make_shared<v8::Slice>(padded_sources,
                                                     v0::Constant::create(element::i64, Shape{1}, {0}),
                                                     std::make_shared<v0::Unsqueeze>(numel, zero),
                                                     v0::Constant::create(element::i64, Shape{1}, {1}));
    const auto is_written = std::make_shared<v1::GreaterEqual>(sources, zero);
    const auto flat_value = std::make_shared<v1::Reshape>(
        std::make_shared<v3::Broadcast>(std::make_shared<v1::ConvertLike>(value, base),
                                        std::make_shared<v3::ShapeOf>(positions, element::i64)),
        flat_shape,
        false);
    // Gather zero-fills out of range indices, which are discarded by Select.
    const auto written = std::make_shared<v8::Gather>(flat_value, sources, zero);
    const auto updated =
        std::make_shared<v1::Select>(is_written, written, std::make_shared<v1::Reshape>(base, flat_shape, false));
    return std::make_shared<v1::Reshape>(updated, shape, false);
}

Output<Node> TranslateSession::rebase_alias(const AliasInfo& alias_info, const Output<Node>& new_base) {
    const auto positions = alias_info.positions->get();
    if (!positions.get_node() || is_structured_value(new_base) || is_structured_value(alias_info.output)) {
        return make_unsupported_alias_node(alias_info.decoder,
                                           new_base,
                                           "Cannot update a view returned from " + alias_info.decoder->get_op_type() +
                                               " after its base tensor was mutated.");
    }
    // Gather zero-fills out of range indices, so -1 positions are safe and replaced by Select.
    const auto gathered = std::make_shared<v8::Gather>(
        std::make_shared<v1::Reshape>(new_base, v0::Constant::create(element::i64, Shape{1}, {-1}), false),
        to_padded_index(positions, get_numel(new_base)),
        v0::Constant::create(element::i64, Shape{}, {0}));
    const auto is_missing = std::make_shared<v1::Less>(positions, v0::Constant::create(element::i64, Shape{}, {0}));
    return std::make_shared<v1::Select>(is_missing,
                                        std::make_shared<v1::ConvertLike>(alias_info.output, gathered),
                                        gathered);
}

}  // namespace ov::frontend::pytorch
