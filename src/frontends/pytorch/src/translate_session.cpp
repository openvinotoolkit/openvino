// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "translate_session.hpp"

#include "input_model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/util/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

using namespace ov::op;

TranslateSession::TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                                   const std::map<std::string, PytorchCreatorFunction>& translator_map)
    : m_input_model(input_model),
      m_translator_map(translator_map),
      m_ov_model(nullptr) {}

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
    return convert_pytorch_model(pytorch_model->m_model_decoder, {}, pytorch_model->m_descriptors);
}

std::shared_ptr<Model> TranslateSession::convert_pytorch_model(
    std::shared_ptr<TorchDecoder> pytorch_model,
    const TensorMap& external_tensor_map,
    const std::unordered_map<size_t, PlaceDesc>& external_descriptors) {
    std::shared_ptr<Model> resulting_model;  // define here to make a conversion in a nested scope
    {
        ParameterVector parameters;
        TensorMap tensor_map;  // tensor map of the current context
        std::set<size_t> mutated_tensors;

        //  Go over all pytorch_model inputs and register them in the tensor map:
        auto inputs = pytorch_model->inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::shared_ptr<Node> input_node;
            element::Type type = element::dynamic;
            PartialShape pshape;
            auto desc = external_descriptors.find(inputs[i]);
            if (desc != external_descriptors.end()) {
                if (desc->second.m_value) {
                    input_node = desc->second.m_value;
                } else {
                    pshape = desc->second.m_pshape;
                    type = desc->second.m_type;
                }
            } else {
                pshape = pytorch_model->get_input_shape(i);
                auto type_any = simplified_type_interpret(pytorch_model->get_input_type(i));
                // TODO: Use special API to set custom type specification
                if (type_any.is<element::Type>()) {
                    type = type_any.as<element::Type>();
                }
            }
            if (!input_node) {
                auto parameter = std::make_shared<v0::Parameter>(type, pshape);
                encode_tensor_name(parameter->output(0), inputs.at(i), pytorch_model->get_input_debug_name(i));
                parameters.push_back(parameter);
                input_node = parameter;
                auto order = pytorch_model->get_input_transpose_order(i);
                if (order.size() > 0 && !std::is_sorted(order.begin(), order.end())) {
                    FRONT_END_GENERAL_CHECK(pshape.is_static(), "Shape must be static.");  // TODO: make dynamic
                    auto sh = pshape.get_shape();
                    Shape new_shape(sh.size());
                    for (size_t i = 0; i < sh.size(); i++) {
                        new_shape[order[i]] = sh[i];
                    }
                    auto shape_const = v0::Constant::create(element::i32, {new_shape.size()}, new_shape);
                    auto reshape = std::make_shared<v1::Reshape>(parameter, shape_const, false);
                    auto order_const = v0::Constant::create(element::i32, {order.size()}, order);
                    auto transpose = std::make_shared<v1::Transpose>(reshape, order_const);
                    input_node = transpose;
                }
            }
            tensor_map[inputs.at(i)] = input_node;
        }

        auto node_visitor = [&](std::shared_ptr<TorchDecoder> node) {
            // Explore all inputs of node. Node may refer to input value that hasn't been created in the current scope.
            // But this value can be found in the outer scope, for this purpose we create new input for the model to
            // link with external scope on a higher level.

            auto raw_inputs = node->inputs();
            for (size_t i = 0; i < raw_inputs.size(); ++i) {
                auto input = raw_inputs.at(i);
                if (tensor_map.find(input) == tensor_map.end()) {
                    // Input refers value in the outer scope, need to create a new Parameter in the current scope
                    // Linkage to external scope will be performed on the level of the parent operation (if or loop)
                    // TODO: Eliminate duplication with the main code for Parameters creation
                    PartialShape ps = node->get_input_shape(i);
                    auto type = simplified_type_interpret(node->get_input_type(i));
                    // TODO: Use special API to set custom type specification
                    auto parameter = std::make_shared<v0::Parameter>(element::dynamic, ps);
                    // TODO: Missing get_input_transpose_order handling for not trivial layouts
                    tensor_map[input] = parameter;
                    // set name of parameter to the index of node in the model
                    encode_tensor_name(parameter->output(0), input);
                    parameters.push_back(parameter);
                }
            }
            auto context = NodeContext(node, &tensor_map, &parameters, external_tensor_map, this);
            auto converted_outputs = convert_node(context);

            auto mutated_t = context.get_mutated_tensors();
            mutated_tensors.insert(mutated_t.begin(), mutated_t.end());

            auto fw_outputs = node->outputs();
            // Ops with subgraphs or with mutated inputs may have more outputs after conversion compared to pytorch ones
            FRONT_END_OP_CONVERSION_CHECK(fw_outputs.size() <= converted_outputs.size(),
                                          "Number of ",
                                          node->get_op_type(),
                                          " outputs greater then number of converted outputs.");

            // TODO: Make sure that mapping of fw_outputs to converted_outputs does always work
            // FIXME: Now it is not true for at least prim::Constant
            for (size_t i = 0; i < fw_outputs.size(); ++i) {
                size_t fw_tensor_id = node->output(i);
                FRONT_END_GENERAL_CHECK(tensor_map.find(fw_tensor_id) == tensor_map.end(),
                                        "Duplicated producer for PT value with unique ID: ",
                                        fw_tensor_id);
                tensor_map[fw_tensor_id] = converted_outputs[i];
                encode_tensor_name(converted_outputs[i], fw_tensor_id, node->get_output_debug_name(i));
            }
        };

        FRONT_END_GENERAL_CHECK(pytorch_model->get_subgraph_size() == 1, "Model should have exactly 1 subgraph.");
        pytorch_model->visit_subgraph(node_visitor);

        ResultVector results;
        for (size_t i = 0; i < pytorch_model->num_of_outputs(); ++i) {
            size_t id = pytorch_model->output(i);
            if (tensor_map.find(id) == tensor_map.end()) {
                // Not found in this scope, adding Parameter to connect to external scope
                auto parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                encode_tensor_name(parameter->output(0), id);
                parameters.push_back(parameter);
                tensor_map[id] = parameter;
            }
            auto ov_output = tensor_map[id];
            auto order = pytorch_model->get_output_transpose_order(i);
            FRONT_END_GENERAL_CHECK(order.size() == 0 || std::is_sorted(order.begin(), order.end()),
                                    "Output strides have wrong order.");
            FRONT_END_GENERAL_CHECK(ov_output.get_names().size() > 0,
                                    "Tensor doesn't have name, while it should have name: ",
                                    id);
            auto result = std::make_shared<v0::Result>(ov_output);
            results.push_back(result);
        }

        // Since parameters can be added we need to list all current parameters
        std::set<size_t> param_names;
        for (const auto& param : parameters) {
            auto input_idx = decode_tensor_name(param->output(0));
            param_names.insert(input_idx);
        }
        for (const auto& tensor_id : mutated_tensors) {
            if (param_names.count(tensor_id)) {
                FRONT_END_GENERAL_CHECK(tensor_map.count(tensor_id),
                                        "Tensor with id: ",
                                        tensor_id,
                                        " doesn't exist in tensor map.");
                // model input was mutated we need to make a result for it
                auto mutated_tensor = tensor_map.at(tensor_id);
                // empty external_tensor_map means this is main body of the model and we don't want to create
                // additional outputs in that case.
                if (mutated_tensor.get_target_inputs().empty() && !external_tensor_map.empty())
                    results.push_back(std::make_shared<v0::Result>(tensor_map.at(tensor_id)));
            }
        }
        resulting_model = std::make_shared<Model>(results, parameters);
        // Did a conversion in a nested scope to automatically remove any holders of nodes except those in the graph
    }

    return resulting_model;
}

OutputVector TranslateSession::convert_node(NodeContext& context) {
    try {
        auto it = m_translator_map.find(context.get_op_type());
        if (it != m_translator_map.end()) {
            return it->second(context);
        }

    } catch (std::runtime_error& e) {
        OPENVINO_DEBUG << "Exception happened during conversion of op: " << context.get_op_type()
                       << " with schema: " << context.get_schema() << ": " << e.what() << '\n';
    } catch (...) {
        OPENVINO_DEBUG << "Some exception happened during conversion of node of type: " << context.get_op_type()
                       << '\n';
    }
    // Create PtFrameworkNode for everything that wasn't able to be converted normally
    return make_framework_node(context);
}

void TranslateSession::encode_tensor_name(Output<Node> output, size_t tensor_idx, std::string debug_name) {
    if (!output.get_names().empty()) {
        OPENVINO_DEBUG << "Tensor names already exist: " << output.get_any_name() << ". Rewriting with " << tensor_idx;
    }
    auto has_dname = !debug_name.empty();
    auto name = std::to_string(tensor_idx);
    if (has_dname && name == debug_name)
        has_dname = false;

    if (m_counter_map.count(tensor_idx)) {
        auto&& pair = m_counter_map[tensor_idx];
        auto new_name = name + '_' + std::to_string(++pair.first);
        pair.second.set_names({new_name});
        pair.second = output;
        if (has_dname) {
            output.set_names({name, debug_name});
        } else {
            output.set_names({name});
        }
    } else {
        m_counter_map[tensor_idx] = {0, output};
        if (has_dname) {
            output.set_names({name, debug_name});
        } else {
            output.set_names({name});
        }
    }
}

size_t TranslateSession::decode_tensor_name(const Output<Node>& output) {
    // any_name should always return numerical value even if there is a word value exist in names
    const auto& name = output.get_any_name();
    // numbers after "_" will be ignored by stoll function
    return static_cast<size_t>(std::stoll(name));
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
