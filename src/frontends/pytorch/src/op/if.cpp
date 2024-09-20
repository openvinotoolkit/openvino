// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/util/log.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
// TODO: Ticket 106627. This is a WA and will work only if both branches of if will eventually go to the operation that
// will have same output type for both types
void align_result_types(const NodeContext& context, std::shared_ptr<v0::Result> r1, std::shared_ptr<v0::Result> r2) {
    auto r1_tensor = r1->input_value(0);
    auto r2_tensor = r2->input_value(0);
    auto r1_type = r1_tensor.get_element_type();
    auto r2_type = r2_tensor.get_element_type();
    if (r1_type == r2_type)
        return;
    element::Type merged_type;
    if (element::Type::merge(merged_type, r1_type, r2_type)) {
        if (r1_type != merged_type) {
            auto convert1 = std::make_shared<v0::Convert>(r1_tensor, merged_type);
            r1->set_argument(0, convert1);
        }
        if (r2_type != merged_type) {
            auto convert2 = std::make_shared<v0::Convert>(r2_tensor, merged_type);
            r2->set_argument(0, convert2);
        }
    } else {
        if (r1_type.bitwidth() >= r2_type.bitwidth()) {
            auto convert = std::make_shared<v0::Convert>(r2_tensor, r1_type);
            r2->set_argument(0, convert);
        } else {
            auto convert = std::make_shared<v0::Convert>(r1_tensor, r2_type);
            r1->set_argument(0, convert);
        }
    }
}
}  // namespace

OutputVector translate_if(const NodeContext& context) {
    auto if_node = std::make_shared<v8::If>(context.get_input(0));
    context.mark_node(if_node);
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 2, "If must have 2 subgraphs.");

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
    auto session = context.get_session();
    for (const auto& param : then_body->get_parameters()) {
        auto input_idx = session->decode_tensor_name(param->output(0));
        PYTORCH_OP_CONVERSION_CHECK(inputs_map.count(input_idx) == 0,
                                    "More than one then_body input with same tensor name: ",
                                    input_idx,
                                    "; existing: ",
                                    inputs_map.at(input_idx)[0],
                                    " adding: ",
                                    param);
        inputs_map[input_idx] = {param, nullptr};
    }
    for (const auto& param : else_body->get_parameters()) {
        auto input_idx = session->decode_tensor_name(param->output(0));
        if (inputs_map.count(input_idx)) {
            inputs_map[input_idx][1] = param;
        } else {
            inputs_map[input_idx] = {nullptr, param};
        }
    }
    OutputVector res;
    const auto num_outs = context.get_output_size();
    const auto& then_results = then_body->get_results();
    const auto& else_results = else_body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(then_results.size() >= num_outs && else_results.size() >= num_outs,
                                "Else or then body have less outputs than prim::If requires.");
    for (size_t i = 0; i < num_outs; i++) {
        align_result_types(context, then_results[i], else_results[i]);
        res.push_back(if_node->set_output(then_results[i], else_results[i]));
    }
    // Each body can have mutated outputs that are not included into pytorch node outputs.
    std::map<size_t, std::shared_ptr<v0::Result>> extra_then_body_results;
    std::map<size_t, std::shared_ptr<v0::Result>> extra_else_body_results;
    std::set<size_t> extra_output_idxs;
    for (size_t i = num_outs; i < then_results.size(); i++) {
        const auto result = then_results[i];
        auto output_idx = session->decode_tensor_name(result->input(0).get_source_output());
        PYTORCH_OP_CONVERSION_CHECK(extra_then_body_results.count(output_idx) == 0,
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
        auto output_idx = session->decode_tensor_name(result->input(0).get_source_output());
        PYTORCH_OP_CONVERSION_CHECK(extra_else_body_results.count(output_idx) == 0,
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
            std::shared_ptr<v0::Parameter> new_parameter;
            if (inputs_map.count(output_idx) && inputs_map[output_idx][0]) {
                // parameter already exist in inputs
                new_parameter = inputs_map[output_idx][0];
            } else {
                new_parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                session->encode_tensor_name(new_parameter->output(0), output_idx);
                then_body->add_parameters({new_parameter});
                PYTORCH_OP_CONVERSION_CHECK(inputs_map.count(output_idx),
                                            "Input must exist in then body: ",
                                            output_idx);
                inputs_map[output_idx][0] = new_parameter;
            }
            auto new_result = std::make_shared<v0::Result>(new_parameter);
            then_body->add_results({new_result});
            then_body->validate_nodes_and_infer_types();
            extra_then_body_results[output_idx] = new_result;
            OPENVINO_DEBUG("Modified then body: ", if_node, "\n");
        } else if (!extra_else_body_results.count(output_idx)) {
            // Need to add Parameter->Result construction in else body
            std::shared_ptr<v0::Parameter> new_parameter;
            if (inputs_map.count(output_idx) && inputs_map[output_idx][1]) {
                // parameter already exist in inputs
                new_parameter = inputs_map[output_idx][1];
            } else {
                new_parameter = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic());
                session->encode_tensor_name(new_parameter->output(0), output_idx);
                else_body->add_parameters({new_parameter});
                PYTORCH_OP_CONVERSION_CHECK(inputs_map.count(output_idx),
                                            "Input must exist in then body: ",
                                            output_idx);
                inputs_map[output_idx][1] = new_parameter;
            }
            auto new_result = std::make_shared<v0::Result>(new_parameter);
            else_body->add_results({new_result});
            else_body->validate_nodes_and_infer_types();
            extra_else_body_results[output_idx] = new_result;
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
        align_result_types(context, extra_then_body_results.at(output_idx), extra_else_body_results.at(output_idx));
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