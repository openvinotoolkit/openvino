// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "helper_ops/gather_assign.hpp"
#include "helper_ops/slice_assign.hpp"
#include "input_model.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/log.hpp"
#include "place.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

namespace {
// FX node IDs are local to each GraphModule, including control-flow bodies.
class AliasScope {
public:
    AliasScope(TranslateSession& session, bool enabled) : m_session(session), m_enabled(enabled) {
        if (m_enabled) {
            m_aliases.swap(m_session.m_may_be_alias);
        }
    }

    ~AliasScope() {
        if (m_enabled) {
            m_aliases.swap(m_session.m_may_be_alias);
        }
    }

private:
    TranslateSession& m_session;
    bool m_enabled;
    decltype(TranslateSession::m_may_be_alias) m_aliases;
};

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
    AliasScope alias_scope(*this, pytorch_model->decoder_type_name() == "fx");
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
            const bool has_out = node->decoder_type_name() == "fx" && context.has_attribute("out");
            if (has_out) {
                FRONT_END_OP_CONVERSION_CHECK(converted_outputs.size() == 1, "Expected a single out tensor.");
                const auto out = context.get_input("out");
                converted_outputs[0] = ComplexTypeMark::convert_like(context, converted_outputs[0], out);
                context.mutate_input("out", converted_outputs[0]);
            }

            const auto& fw_outputs = node->outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after conversion compared to pytorch ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          context.get_op_type(),
                                          " outputs greater than number of converted outputs, which are",
                                          fw_outputs.size(),
                                          " and ",
                                          converted_outputs.size(),
                                          " respectively.");

            const bool has_inputs = !node->inputs().empty();
            const size_t in_tensor_id =
                has_out ? node->get_named_input("out") : (has_inputs ? node->inputs().at(0) : 0);
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                if (has_out || (has_inputs && node->may_produce_alias(0, i))) {
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
                    m_may_be_alias[fw_tensor_id] = {
                        in_tensor_id,
                        node,
                        converted_outputs[i],
                        node->decoder_type_name() == "fx" ? tensor_map->at(in_tensor_id) : Output<Node>{}};
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
                (*tensor_map)[fw_tensor_id] = converted_outputs[i];
                encode_tensor_name(converted_outputs[i], fw_tensor_id, {node->get_output_debug_name(i)});
            }
        };

        FRONT_END_GENERAL_CHECK(pytorch_model->decoder_type_name() != "ts" || pytorch_model->get_subgraph_size() == 1,
                                "Model should have exactly 1 subgraph for TorchScript.");
        pytorch_model->visit_subgraph(node_visitor);

        std::unique_ptr<NodeContext> output_context;
        if (pytorch_model->decoder_type_name() == "fx") {
            output_context = std::make_unique<NodeContext>(pytorch_model,
                                                           external_tensor_map,
                                                           tensor_map,
                                                           parameters,
                                                           mutated_tensors,
                                                           this);
        }
        ResultVector results;
        if (input_model) {
            // For the case when we have InputModel we need to have same order as its outputs
            for (auto& output_p : input_model->get_outputs()) {
                auto pytorch_place = std::dynamic_pointer_cast<pytorch::Place>(output_p);
                FRONT_END_GENERAL_CHECK(pytorch_place, "Only place produced by PyTorch Frontend is supported.");
                auto tensor_id = pytorch_place->get_tensor_index();
                auto ov_output = output_context ? output_context->resolve_tensor(tensor_id) : tensor_map->at(tensor_id);
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
                auto result =
                    std::make_shared<v0::Result>(output_context ? output_context->resolve_tensor(id) : it->second);
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
        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    return resulting_model;
}

OutputVector TranslateSession::convert_node(const NodeContext& context) {
    std::string exception;
    try {
        auto op_type = context.get_op_type();
        auto it = m_translator_map.find(op_type);
        if (it == m_translator_map.end() && op_type.find("aten.") == 0) {
            const auto overload = op_type.find('.', 5);
            op_type = "aten::" + op_type.substr(5, overload - 5);
            it = m_translator_map.find(op_type);
        }
        if (it != m_translator_map.end()) {
            auto outputs = it->second(context);
            // FX represents multiple operator results as a single tuple value.
            if (context.get_decoder()->decoder_type_name() == "fx" && outputs.size() > 1) {
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

using ReversepropCreatorFunction = std::function<ov::Output<ov::Node>(const Output<Node>&, const Output<Node>&)>;

Output<Node> TranslateSession::get_reverseprop_op(const std::shared_ptr<TorchDecoder>& node,
                                                  const Output<Node>& direct_op_output,
                                                  const Output<Node>& value,
                                                  const Output<Node>& base) {
    static const std::map<std::string, ReversepropCreatorFunction> backprop_map = {
        {"aten::slice", slice_reverseprop},
        {"aten::select", select_reverseprop},
    };

    try {
        const auto direct_node = direct_op_output.get_node_shared_ptr();
        if (node->decoder_type_name() == "fx") {
            const auto& name = node->get_op_type();
            if (name.find("aten.split.") == 0 || name.find("aten.chunk.") == 0 ||
                name.find("aten.unsafe_chunk.") == 0 || name.find("aten.tensor_split.") == 0 ||
                name.find("aten.split_with_sizes.") == 0 || name.find("aten.unbind.") == 0) {
                // getitem has already scattered the modified element into the base.
                return value;
            }
            if (base.get_node()) {
                if (direct_op_output == base) {
                    return value;
                }
                if (ov::is_type<SequenceMark>(base.get_node_shared_ptr())) {
                    for (const auto& element : base.get_node_shared_ptr()->input_values()) {
                        if (direct_op_output == element) {
                            return get_reverseprop_op(node, direct_op_output, value);
                        }
                    }
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
                const auto order = ov::util::get_constant_from_source(direct_node->input_value(1));
                FRONT_END_OP_CONVERSION_CHECK(order, "Cannot reverse a view with a dynamic permutation.");
                const auto axes = order->cast_vector<int64_t>();
                std::vector<int64_t> inverse(axes.size());
                for (size_t i = 0; i < axes.size(); ++i) {
                    inverse.at(axes[i]) = i;
                }
                return std::make_shared<v1::Transpose>(value,
                                                       v0::Constant::create(element::i64, Shape{axes.size()}, inverse));
            }
        }
        auto it = backprop_map.find(node->get_op_type());
        if (it != backprop_map.end()) {
            return it->second(direct_op_output, value);
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

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
