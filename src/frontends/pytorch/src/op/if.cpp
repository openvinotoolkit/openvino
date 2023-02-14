// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/util/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_if(NodeContext& context) {
    auto if_node = std::make_shared<opset10::If>(context.get_input(0));
    context.mark_node(if_node);
    auto decoder = context.get_decoder();
    FRONT_END_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 2, "If must have 2 subgraphs.");

    auto then_decoder = decoder->get_subgraph_decoder(0);
    auto then_body = context.convert_subgraph(0);
    if_node->set_then_body(then_body);
    auto then_inputs = then_decoder->inputs();

    auto else_decoder = decoder->get_subgraph_decoder(1);
    auto else_body = context.convert_subgraph(1);
    if_node->set_else_body(else_body);
    auto else_inputs = else_decoder->inputs();

    std::set<size_t> input_idxs;
    input_idxs.insert(then_inputs.begin(), then_inputs.end());
    input_idxs.insert(else_inputs.begin(), else_inputs.end());

    std::map<size_t, ParameterVector> inputs_map;
    std::map<size_t, ResultVector> outputs_map;
    for (const auto& param : then_body->get_parameters()) {
        auto name = param->get_output_tensor(0).get_any_name();
        size_t input_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(input_idx) == 0,
                                      "More than one then_body input with same tensor name: ",
                                      input_idx,
                                      "; existing: ",
                                      inputs_map.at(input_idx)[0],
                                      " adding: ",
                                      param);
        inputs_map[input_idx] = {param, nullptr};
    }
    for (const auto& param : else_body->get_parameters()) {
        auto name = param->get_output_tensor(0).get_any_name();
        size_t input_idx = (size_t)std::stoll(name);
        if (inputs_map.count(input_idx)) {
            inputs_map[input_idx][1] = param;
        } else {
            inputs_map[input_idx] = {nullptr, param};
        }
    }
    OutputVector res;
    const auto num_outs = context.get_output_size();
    const auto then_results = then_body->get_results();
    const auto else_results = else_body->get_results();
    FRONT_END_OP_CONVERSION_CHECK(then_results.size() >= num_outs && else_results.size() >= num_outs,
                                  "Else or then body have less outputs than prim::If requires.");
    for (size_t i = 0; i < num_outs; i++) {
        res.push_back(if_node->set_output(then_results[i], else_results[i]));
    }
    // Each body can have mutated outputs that are not included into pytorch node outputs.
    std::map<size_t, std::shared_ptr<opset10::Result>> extra_then_body_results;
    std::map<size_t, std::shared_ptr<opset10::Result>> extra_else_body_results;
    std::set<size_t> extra_output_idxs;
    for (size_t i = num_outs; i < then_results.size(); i++) {
        const auto result = then_results[i];
        const auto name = result->input(0).get_tensor().get_any_name();
        size_t output_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(extra_then_body_results.count(output_idx) == 0,
                                      "More than one then_body output with same tensor name: ",
                                      output_idx,
                                      "; existing: ",
                                      extra_then_body_results.at(output_idx),
                                      " adding: ",
                                      result);
        extra_then_body_results[output_idx] = result;
        extra_output_idxs.insert(output_idx);
    }
    for (size_t i = num_outs; i < else_results.size(); i++) {
        const auto result = else_results[i];
        const auto name = result->input(0).get_tensor().get_any_name();
        size_t output_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(extra_else_body_results.count(output_idx) == 0,
                                      "More than one else_body output with same tensor name: ",
                                      output_idx,
                                      "; existing: ",
                                      extra_else_body_results.at(output_idx),
                                      " adding: ",
                                      result);
        extra_else_body_results[output_idx] = result;
        extra_output_idxs.insert(output_idx);
    }
    // Each extra output may not have same extra output in the other body, so we need to create Parameter->Result
    // pattern in the body.
    for (const auto& output_idx : extra_output_idxs) {
        if (!extra_then_body_results.count(output_idx)) {
            // Need to add Parameter->Result construction in then body
            auto new_parameter = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
            new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
            auto new_result = std::make_shared<opset10::Result>(new_parameter);
            then_body->add_parameters({new_parameter});
            then_body->add_results({new_result});
            then_body->validate_nodes_and_infer_types();
            FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in else body");
            inputs_map[output_idx][0] = new_parameter;
            extra_then_body_results[output_idx] = new_result;
            OPENVINO_DEBUG << "Modified then body: " << if_node << '\n';
        } else if (!extra_else_body_results.count(output_idx)) {
            // Need to add Parameter->Result construction in else body
            auto new_parameter = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
            new_parameter->get_output_tensor(0).add_names({std::to_string(output_idx)});
            auto new_result = std::make_shared<opset10::Result>(new_parameter);
            else_body->add_parameters({new_parameter});
            else_body->add_results({new_result});
            else_body->validate_nodes_and_infer_types();
            FRONT_END_OP_CONVERSION_CHECK(inputs_map.count(output_idx), "Input must exist in then body");
            inputs_map[output_idx][1] = new_parameter;
            extra_else_body_results[output_idx] = new_result;
            OPENVINO_DEBUG << "Modified else body: " << if_node << '\n';
        }
    }
    // Create prim::If inputs and outputs
    for (const auto& input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context.get_tensor_from_model_or_create_input(input.first);
            if_node->set_input(external_output, input.second[0], input.second[1]);
        } else {
            auto external_output = context.get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                if_node->set_input(external_output, input.second[0], input.second[1]);
            }
        }
    }
    for (const auto& output_idx : extra_output_idxs) {
        context.add_tensor_to_context(
            output_idx,
            if_node->set_output(extra_then_body_results.at(output_idx), extra_else_body_results.at(output_idx)));
    }
    if_node->validate_and_infer_types();
    return res;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov