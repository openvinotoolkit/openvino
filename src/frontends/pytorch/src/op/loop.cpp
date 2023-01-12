// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_loop(NodeContext& context) {
    auto loop = std::make_shared<opset8::Loop>(context.get_input(0), context.get_input(1));
    auto decoder = context.get_decoder();
    OV_FRONTEND_REQUIRE(decoder->get_subgraph_size() == 1);
    auto subgraph_decoder = decoder->get_subgraph_decoder(0);
    auto body = context.convert_subgraph(0);
    loop->set_function(body);
    opset8::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    auto inputs = subgraph_decoder->inputs();
    std::set<size_t> input_idxs(inputs.begin(), inputs.end());
    std::map<size_t, ParameterVector> inputs_map;

    auto body_parameters = body->get_parameters();
    // #0 parameter is counter
    for (int i = 1; i < body_parameters.size(); i++) {
        auto param = body_parameters[i];
        auto name = param->get_output_tensor(0).get_any_name();
        size_t input_idx = (size_t)std::stoll(name);
        if (inputs_map.count(input_idx)) {
            inputs_map[input_idx] = {param};
        } else {
            inputs_map[input_idx].push_back(param);
        }
    }
    for (const auto& input : inputs_map) {
        if (!input_idxs.count(input.first)) {
            auto external_output = context.get_tensor_from_model_or_create_input(input.first);
            loop->set_invariant_inputs(external_output, input.second);
        } else {
            auto external_output = context.get_tensor_from_model(input.first);
            if (external_output.get_node()) {
                loop->set_invariant_inputs(external_output, input.second);
            }
        }
    }
    // TODO: Connect back edges (merged inputs)
    auto body_results = body->get_results();
    FRONT_END_OP_CONVERSION_CHECK(body_results.size() > 0, "At least one output from loop is required - condition.");
    std::set<size_t> output_idxs;
    // 0 output is condition, do not need to connect it
    for (int i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto name = result->input(0).get_tensor().get_any_name();
        size_t out_idx = (size_t)std::stoll(name);
        FRONT_END_OP_CONVERSION_CHECK(output_idxs.count(out_idx) == 0,
                                      "More then one body output with same tensor name.");
        output_idxs.insert(out_idx);
        context.add_tensor_to_context(out_idx, loop->get_iter_value(result, -1));
    }
    loop->validate_and_infer_types();
    return {context.mark_node(loop)->outputs()};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov