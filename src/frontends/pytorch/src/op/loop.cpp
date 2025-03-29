// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_loop(const NodeContext& context) {
    const auto& inputs = context.inputs();
    PYTORCH_OP_CONVERSION_CHECK(inputs.size() >= 2, "Loop must have at least 2 inputs.");
    auto loop = std::make_shared<ov::op::v5::Loop>(inputs[0], inputs[1]);
    auto decoder = context.get_decoder();
    PYTORCH_OP_CONVERSION_CHECK(decoder->get_subgraph_size() == 1, "Loop must have 1 subgraph.");
    auto subgraph_decoder = decoder->get_subgraph_decoder(0);
    auto body = context.convert_subgraph(0);
    loop->set_function(body);
    ov::op::v5::Loop::SpecialBodyPorts spec_ports{0, 0};
    loop->set_special_body_ports(spec_ports);

    // process outputs first
    auto session = context.get_session();
    auto body_results = body->get_results();
    PYTORCH_OP_CONVERSION_CHECK(body_results.size() > 0, "At least one output from loop is required - condition.");
    std::map<size_t, Output<Node>> output_idxs;
    // 0 output is condition, do not need to connect it
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        PYTORCH_OP_CONVERSION_CHECK(output_idxs.count(out_idx) == 0,
                                    "More then one body output with same tensor name.");
        output_idxs[out_idx] = result;
    }

    auto body_parameters = body->get_parameters();
    // #0 body parameter is counter;
    PYTORCH_OP_CONVERSION_CHECK(body_parameters.size() > 0, "At least one input to Loop body is required");
    // Set counter shape
    body_parameters[0]->set_partial_shape(PartialShape{});
    // #0 loop input is  trip_count, #1 loop input is condition
    // Connect other inputs
    for (size_t i = 2; i < inputs.size(); i++) {
        if (i <= subgraph_decoder->num_of_outputs()) {
            loop->set_merged_input(body_parameters[i - 1], inputs[i], body_results[i - 1]);
        } else {
            loop->set_invariant_input(body_parameters[i - 1], inputs[i]);
        }
    }
    // Connect inputs from external context
    for (auto i = inputs.size() - 1; i < body_parameters.size(); i++) {
        auto param = body_parameters[i];
        auto input_idx = session->decode_tensor_name(param->output(0));
        auto external_output = context.get_tensor_from_model_or_create_input(input_idx);
        if (output_idxs.count(input_idx)) {
            loop->set_merged_input(param, external_output, output_idxs.at(input_idx));
        } else {
            loop->set_invariant_input(param, external_output);
        }
    }

    // connect outputs
    for (size_t i = 1; i < body_results.size(); i++) {
        auto result = body_results[i];
        auto out_idx = session->decode_tensor_name(result->input(0).get_source_output());
        context.add_tensor_to_context(out_idx, loop->get_iter_value(result, -1));
    }
    loop->validate_and_infer_types();
    return {context.mark_node(loop)->outputs()};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
